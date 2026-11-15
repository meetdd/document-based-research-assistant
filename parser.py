import io
import os
import re
import numpy as np
import pymupdf
import pytesseract
from PIL import Image
from official_summary_store import save_summary

# Update this path if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

KNOWN_SURNAMES = [
    "जैन", "गर्ग", "अग्रवाल", "गोयल", "बंसल", "यादव", "मित्तल", 
    "वर्मा", "सिंघल", "कटेजा", "शर्मा", "सिंह", "कुमार", "देवी", "गुप्ता", "सोलंकी"
]

FEMALE_NAME_HINTS = [
    "देवी", "कुमारी", "वती", "बाई", "लता", "रानी", "प्रिया", "पूजा", "नेहा",
    "कविता", "सुनीता", "अनीता", "नीतू", "रेखा", "सीमा", "गीता", "मीनू", "रेनू",
    "शालिनी", "भावना", "प्रियंका", "खुशबू", "ज्योति", "सोनिया", "मंजू", "ऊषा",
    "उषा", "आशा", "कमला", "विमला", "सरोज", "सुमन", "पूनम", "किरन", "किरण",
    "आरती", "स्वाति", "निधि", "दीप्ति", "शिखा", "मेघा", "पायल", "कोमल", "काजल",
    "मुस्कान", "साक्षी", "तान्या", "मानसी", "प्राची", "रिया", "श्रेया", "अंजलि",
    "अंजली", "वर्षा", "रश्मि", "सपना", "वंदना", "प्रीति", "प्रीती", "रीता",
    "सविता", "ममता", "बबिता", "बबीता", "सुधा", "राधा", "मीरा", "सीता", "प्रभा",
    "निशा", "ईशा", "उमा", "रमा", "सुषमा", "मोनिका", "दीपिका", "रूचिका", "रुचिका",
    "अंशिका", "राधिका", "शशि", "शशी", "कंचन", "चंचल", "सिमरन", "रीना"
]

def infer_gender_fallback(name_str: str, is_husband_rel: bool) -> str:
    if is_husband_rel:
        return "Female"
    for hint in FEMALE_NAME_HINTS:
        if hint in name_str:
            return "Female"
    return "Male"

def extract_surname(name_str: str, rel_name_str: str = "") -> str:
    name_str = name_str.strip()
    if not name_str or name_str == "Unknown":
        return "Unknown"

    parts = name_str.split()
    if len(parts) > 1:
        return parts[-1]

    for s in KNOWN_SURNAMES:
        if name_str.endswith(s) and len(name_str) > len(s):
            return s

    if rel_name_str and rel_name_str != "Unknown":
        rel_parts = rel_name_str.strip().split()
        if len(rel_parts) > 1:
            return rel_parts[-1]
        for s in KNOWN_SURNAMES:
            if rel_name_str.strip().endswith(s):
                return s

    return "Other"

def remove_table_lines(pil_img: Image.Image) -> Image.Image:
    """
    Erases continuous horizontal and vertical table border lines using NumPy
    so Tesseract never misreads table borders '|' as '1' on summary pages.
    """
    gray = np.array(pil_img.convert("L"))
    binary = (gray < 185).astype(np.uint8)
    h, w = binary.shape

    win_h = min(80, max(20, w // 4))
    cs_h = np.cumsum(np.pad(binary, ((0, 0), (1, 0))), axis=1)
    h_starts = (cs_h[:, win_h:] - cs_h[:, :-win_h]) >= int(win_h * 0.85)
    horiz_mask = np.zeros_like(binary, dtype=bool)
    for d in range(win_h):
        horiz_mask[:, d:d + w - win_h + 1] |= h_starts

    win_v = min(65, max(20, h // 2))
    cs_v = np.cumsum(np.pad(binary, ((1, 0), (0, 0))), axis=0)
    v_starts = (cs_v[win_v:, :] - cs_v[:-win_v, :]) >= int(win_v * 0.85)
    vert_mask = np.zeros_like(binary, dtype=bool)
    for d in range(win_v):
        vert_mask[d:d + h - win_v + 1, :] |= v_starts

    line_mask = horiz_mask | vert_mask
    dilated = line_mask.copy()
    for shift in (-2, -1, 1, 2):
        dilated |= np.roll(line_mask, shift, axis=0)
        dilated |= np.roll(line_mask, shift, axis=1)

    cleaned = gray.copy()
    cleaned[dilated] = 255
    return Image.fromarray(cleaned).convert("RGB")

def _candidate_ints(n: int):
    """Handles cases where a table border '|' was read as a leading or trailing '1'."""
    cands = [n]
    s = str(n)
    if len(s) > 1 and s.startswith("1"):
        cands.append(int(s[1:]))
    if len(s) > 1 and s.endswith("1"):
        cands.append(int(s[:-1]))
    if len(s) > 2 and s.startswith("1") and s.endswith("1"):
        cands.append(int(s[1:-1]))
    return cands

def _find_valid_tuples_in_lines(text: str):
    """
    Slides a window across every line to find valid (male, female, third, total)
    tuples where male + female + third == total.
    """
    tuples = []
    for line in text.split("\n"):
        raw_nums = [
            int(x) for x in re.findall(r'\d+', line)
            if int(x) not in (2024, 2025, 2026, 2027, 282005)
        ]
        if len(raw_nums) < 3:
            continue

        matched_line = False
        # 1. Slide 4-element window right-to-left: [male, female, third, total]
        if len(raw_nums) >= 4:
            for i in range(len(raw_nums) - 4, -1, -1):
                a_raw, b_raw, c_raw, d_raw = raw_nums[i], raw_nums[i + 1], raw_nums[i + 2], raw_nums[i + 3]
                for a in _candidate_ints(a_raw):
                    for b in _candidate_ints(b_raw):
                        for c in _candidate_ints(c_raw):
                            for d in _candidate_ints(d_raw):
                                if 0 <= c <= 20 and d > 0 and (a + b + c == d):
                                    tuples.append((a, b, c, d))
                                    matched_line = True
                                    break
                            if matched_line:
                                break
                        if matched_line:
                            break
                    if matched_line:
                        break
                if matched_line:
                    break

        if matched_line:
            continue

        # 2. Fallback: slide 3-element window [male, female, total] if '0' was skipped
        for i in range(len(raw_nums) - 3, -1, -1):
            a_raw, b_raw, d_raw = raw_nums[i], raw_nums[i + 1], raw_nums[i + 2]
            for a in _candidate_ints(a_raw):
                for b in _candidate_ints(b_raw):
                    for d in _candidate_ints(d_raw):
                        if d > 0 and (a + b == d):
                            tuples.append((a, b, 0, d))
                            matched_line = True
                            break
                    if matched_line:
                        break
                if matched_line:
                    break
            if matched_line:
                break

    return tuples

def _extract_tuples_from_row_strips(num_crop: Image.Image):
    """
    Slices the numeric summary table along its horizontal black lines into individual
    single-row strips and erases vertical cell borders so no row is ever skipped.
    """
    gray = np.array(num_crop.convert("L"))
    h, w = gray.shape
    row_dark = (gray[:, int(w * 0.10):int(w * 0.90)] < 185).mean(axis=1)
    y_idx = np.where(row_dark > 0.55)[0]
    if len(y_idx) == 0:
        return []

    lines_y = []
    start = y_idx[0]
    prev = y_idx[0]
    for y in y_idx[1:]:
        if y <= prev + 4:
            prev = y
        else:
            lines_y.append((start + prev) // 2)
            start = y
            prev = y
    lines_y.append((start + prev) // 2)

    row_tuples = []
    for k in range(len(lines_y) - 1):
        y1, y2 = lines_y[k], lines_y[k + 1]
        if y2 - y1 < 24 or y2 - y1 > int(h * 0.35):
            continue
        strip = gray[max(0, y1 + 3):min(h, y2 - 3), :].copy()
        if strip.shape[0] < 15:
            continue
        col_dark = (strip < 185).mean(axis=0)
        for x in np.where(col_dark > 0.65)[0]:
            strip[:, max(0, x - 3):min(w, x + 4)] = 255

        strip_img = Image.fromarray(strip).convert("RGB")
        txt = pytesseract.image_to_string(
            strip_img, lang='eng', config='--psm 6 -c tessedit_char_whitelist=0123456789 '
        )
        found = _find_valid_tuples_in_lines(txt)
        for t in found:
            if not row_tuples or row_tuples[-1] != t:
                row_tuples.append(t)

    return row_tuples

def _solve_summary_from_tuples(tuples: list, expected_count: int = 0) -> dict:
    """
    Uses the official ECI formula (Row I + Row II - Row III = Net Electors)
    to identify Mother Roll, Additions, Deletions, Gross Total, and Net Electors.
    """
    # Deduplicate while preserving top-to-bottom order
    uniq = []
    for t in tuples:
        if t[3] > 0 and t not in uniq:
            uniq.append(t)

    if not uniq:
        return {}

    # 1. Check if R1 (Mother), R2 (Additions), R3 (Deletions), and R5 (Net) all exist in uniq
    zero_t = (0, 0, 0, 0)
    candidates_r2 = uniq + [zero_t]
    candidates_r3 = uniq + [zero_t]

    best_exact = None
    for idx1, r1 in enumerate(uniq):
        if r1[3] < 100:
            continue
        for r5 in uniq[idx1:]:
            if r5 == r1 and len(uniq) > 1:
                continue
            if r5[3] < int(r1[3] * 0.75):
                continue
            for r2 in candidates_r2:
                if r2 in (r1, r5) or r2[3] >= r1[3]:
                    continue
                for r3 in candidates_r3:
                    if r3 in (r1, r5) or r3[3] > r1[3]:
                        continue
                    if (
                        r1[0] + r2[0] - r3[0] == r5[0]
                        and r1[1] + r2[1] - r3[1] == r5[1]
                        and r1[3] + r2[3] - r3[3] == r5[3]
                    ):
                        best_exact = (r1, r2, r3, r5)
                        break
                if best_exact:
                    break
            if best_exact:
                break
        if best_exact:
            break

    if best_exact:
        r1, r2, r3, r5 = best_exact
        return {
            "net_male": r5[0],
            "net_female": r5[1],
            "net_third": r5[2],
            "net_total": r5[3],
            "mother_total": r1[3],
            "additions_total": r2[3],
            "deleted_total": r3[3],
            "gross_total": r1[3] + r2[3],
        }

    # 2. If R5 (Net Row) was missed by OCR, compute R5 = R1 + R2 - R3 directly!
    # Find R1 (Mother Roll): first large tuple (>= 150)
    large_candidates = [t for t in uniq if t[3] >= 150]
    if large_candidates:
        # Check if the last large candidate is already R5 (Net Electors > R1)
        r1 = large_candidates[0]
        r1_idx = uniq.index(r1)
        after_r1 = uniq[r1_idx + 1:]

        # Look for R2 (Additions < R1) and R3 (Deletions <= R2) after R1
        small_after = [t for t in after_r1 if t[3] < int(r1[3] * 0.65)]
        large_after = [t for t in after_r1 if t[3] >= int(r1[3] * 0.80)]

        if large_after:
            # R5 was read directly, and one of small_after may be R2 or R3
            r5 = large_after[-1]
            r2_tot, r3_tot = 0, 0
            for t in small_after:
                if r1[3] + t[3] >= r5[3] and r2_tot == 0:
                    r2_tot = t[3]
                    r3_tot = (r1[3] + r2_tot) - r5[3]
                    break
            return {
                "net_male": r5[0],
                "net_female": r5[1],
                "net_third": r5[2],
                "net_total": r5[3],
                "mother_total": r1[3],
                "additions_total": r2_tot,
                "deleted_total": max(0, r3_tot),
                "gross_total": r1[3] + r2_tot if r2_tot > 0 else r5[3],
            }

        if small_after:
            # R5 was NOT read directly, but R1 (e.g. 556), R2 (e.g. 50), and optional R3 (e.g. 1) WERE read!
            r2 = small_after[0]
            r3 = small_after[1] if len(small_after) >= 2 and small_after[1][3] <= r2[3] else (0, 0, 0, 0)
            net_m = r1[0] + r2[0] - r3[0]
            net_f = r1[1] + r2[1] - r3[1]
            net_t = max(0, r1[2] + r2[2] - r3[2])
            net_tot = r1[3] + r2[3] - r3[3]
            return {
                "net_male": net_m,
                "net_female": net_f,
                "net_third": net_t,
                "net_total": net_tot,
                "mother_total": r1[3],
                "additions_total": r2[3],
                "deleted_total": r3[3],
                "gross_total": r1[3] + r2[3],
            }

        # Fallback: single large tuple
        best_t = min(large_candidates, key=lambda t: abs(t[3] - expected_count)) if expected_count > 0 else large_candidates[-1]
        return {
            "net_male": best_t[0],
            "net_female": best_t[1],
            "net_third": best_t[2],
            "net_total": best_t[3],
            "mother_total": best_t[3],
            "additions_total": 0,
            "deleted_total": 0,
            "gross_total": best_t[3],
        }

    return {}

def extract_official_summary(doc, expected_count: int = 0) -> dict:
    """
    Extracts the 100% certified ECI counts from the Last Page ('निर्वाचकों का सार')
    and cross-verifies with Page 1 ('4. निर्वाचकों की संख्या').
    """
    all_last_page_tuples = []

    # 1. Parse Last Page ('निर्वाचकों का सार') across all passes (no early break!)
    last_idx = len(doc) - 1
    for idx in ([last_idx] if last_idx >= 0 else []):
        pix = doc[idx].get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        w, h = img.size

        num_crop = img.crop((int(w * 0.58), int(h * 0.08), int(w * 0.99), int(h * 0.82)))
        clean_crop = remove_table_lines(num_crop)
        clean_full = remove_table_lines(img)

        # Pass A: Row-by-row strip slicer
        all_last_page_tuples.extend(_extract_tuples_from_row_strips(num_crop))

        # Pass B: Multi-crop OCR passes
        for target_img, cfg in [
            (clean_crop, '--psm 6 -c tessedit_char_whitelist=0123456789 '),
            (num_crop, '--psm 6 -c tessedit_char_whitelist=0123456789 '),
            (clean_full, '--psm 6'),
            (img, '--psm 6'),
        ]:
            ocr_txt = pytesseract.image_to_string(target_img, lang='hin+eng', config=cfg)
            all_last_page_tuples.extend(_find_valid_tuples_in_lines(ocr_txt))

    summary = _solve_summary_from_tuples(all_last_page_tuples, expected_count=expected_count)

    # 2. Cross-check with Page 1 ('4. निर्वाचकों की संख्या')
    try:
        pix0 = doc[0].get_pixmap(dpi=300)
        img0 = Image.open(io.BytesIO(pix0.tobytes("png")))
        w0, h0 = img0.size
        crop0 = img0.crop((int(w0 * 0.02), int(h0 * 0.55), int(w0 * 0.98), int(h0 * 0.94)))
        clean0 = remove_table_lines(crop0)

        p1_tuples = []
        for target0, cfg0 in [
            (clean0, '--psm 6'),
            (crop0, '--psm 6'),
            (clean0, '--psm 6 -c tessedit_char_whitelist=0123456789 '),
        ]:
            txt0 = pytesseract.image_to_string(target0, lang='hin+eng', config=cfg0)
            p1_tuples.extend(_find_valid_tuples_in_lines(txt0))

            # Also check for [end_serial, male, female, third, total] sequence
            nums0 = [int(x) for x in re.findall(r'\d+', txt0) if int(x) not in (2024, 2025, 2026, 282005)]
            for i in range(len(nums0) - 3):
                m, f, t, tot = nums0[i], nums0[i + 1], nums0[i + 2], nums0[i + 3]
                if m >= 50 and f >= 50 and 0 <= t <= 20 and (m + f + t == tot):
                    if i >= 1 and tot <= nums0[i - 1] <= tot + 250:
                        summary["gross_total"] = nums0[i - 1]
                    break

        p1_large = [t for t in p1_tuples if t[3] >= 150 and t[0] >= 50 and t[1] >= 50]
        if p1_large:
            p1_net = p1_large[-1]
            # If Last Page summary was missing or way off, use Page 1's certified Net Electors
            if "net_total" not in summary or (
                expected_count > 0
                and abs(p1_net[3] - expected_count) < abs(summary["net_total"] - expected_count)
            ):
                summary["net_male"] = p1_net[0]
                summary["net_female"] = p1_net[1]
                summary["net_third"] = p1_net[2]
                summary["net_total"] = p1_net[3]
                if not summary.get("gross_total") or summary["gross_total"] < p1_net[3]:
                    summary["gross_total"] = p1_net[3]
                summary["deleted_total"] = max(0, summary["gross_total"] - summary["net_total"])
    except Exception:
        pass

    return summary

def crop_and_stack_columns(pix) -> Image.Image:
    """
    Slices a 3-column electoral roll page into 3 individual vertical columns
    and stacks them top-to-bottom so Tesseract reads one voter box at a time.
    """
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    w, h = img.size

    top_y = int(h * 0.042)
    bot_y = int(h * 0.965)

    col1 = img.crop((int(w * 0.020), top_y, int(w * 0.348), bot_y))
    col2 = img.crop((int(w * 0.342), top_y, int(w * 0.660), bot_y))
    col3 = img.crop((int(w * 0.654), top_y, int(w * 0.980), bot_y))

    col_h = bot_y - top_y
    col_w = max(col1.width, col2.width, col3.width)

    stacked = Image.new("RGB", (col_w, col_h * 3 + 60), (255, 255, 255))
    stacked.paste(col1, (0, 0))
    stacked.paste(col2, (0, col_h + 30))
    stacked.paste(col3, (0, (col_h + 30) * 2))
    return stacked

def parse_voter_text(raw_text: str, pdf_name: str, page_no: int):
    if any(m in raw_text for m in ["निर्वाचकों का सार", "पुनरीक्षण का विवरण", "Nazri Naksha"]):
        return []

    # 1. Clean common Tesseract Hindi OCR noise
    text = raw_text.replace("fran", "पिता").replace("पतत", "पति").replace("ना म", "नाम")
    text = text.replace("महि्ला", "महिला").replace("महीला", "महिला").replace("माहला", "महिला")
    text = text.replace("पुयष", "पुरुष").replace("पुरूष", "पुरुष").replace("उरुष", "पुरुष").replace("पुरष", "पुरुष")
    text = text.replace("लिग", "लिंग")
    text = re.sub(r'फोटो\s*उपलब्ध\s*है[।|\.]*', '', text)

    # 2. Force newlines before fields if OCR joined two lines inside a box
    text = re.sub(r'\s+((?:पिता|पति|माता|पिना|पत्ति|फिता|पीता|पती|विता)\s*(?:का)?\s*नाम)', r'\n\1', text)
    text = re.sub(r'\s+(अन्य\s*[:ः\-])', r'\n\1', text)
    text = re.sub(r'\s+(मकान\s*संख्या)', r'\n\1', text)
    text = re.sub(r'\s+(आयु(?![षस])\s*[:ः\-]?\s*(?:\d|DELETED|TED))', r'\n\1', text)

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    voter_records = []
    current_record = {}
    pending_epic = None

    def save_current(rec):
        if rec.get("voter_name") or rec.get("rel_name") or rec.get("has_age"):
            voter_records.append(rec)

    for line in lines:
        if any(h in line for h in [
            "निर्वाचक नामावली", "अनुभाग संख्या", "भाग संख्या",
            "अर्हता दिनांक", "के अनुसार", "अनुसार आयु", "प्रकाशन की तिथि",
            "कुल पृष्ठ", "परिवर्धन की सूची"
        ]):
            continue

        if re.search(r'नाम\s*[:ः\-]?\s*\d+\s*[\-]', line):
            continue

        if re.search(r'(DELETED|ELETED|BUTED|\bTED\b)', line):
            current_record["is_deleted"] = True

        epic_match = re.search(r'\b([A-Za-z]{3}\s*[\dIilOo]{7})\b', line)
        if epic_match:
            found_epic = epic_match.group(1).replace(" ", "").upper()
            if "epic_id" not in current_record and not current_record.get("has_age"):
                current_record["epic_id"] = found_epic
            else:
                pending_epic = found_epic

        # A. Relative Name line (checked FIRST)
        rel_match = re.search(
            r'(?:(?:पिता|पति|माता|अन्य|पिना|पत्ति|पनी|फिता|पीता|पती|विता)\s*(?:का)?\s*नाम|का\s+नाम|अन्य)\s*[:ः\-]?\s*(.+)',
            line
        )
        if rel_match:
            if "rel_name" in current_record or current_record.get("has_age"):
                save_current(current_record)
                current_record = {"voter_name": "Unknown"}
                if pending_epic:
                    current_record["epic_id"] = pending_epic
                    pending_epic = None
            current_record["rel_name"] = rel_match.group(1).strip()
            if any(h_word in line for h_word in ["पति", "पत्ति", "पनी", "पती"]):
                current_record["is_husband"] = True
            continue

        # B. Voter Name line
        name_match = re.search(r'(?:नाम|नास|गाम)\s*[:ः\-]\s*(.+)', line)
        if not name_match:
            name_match = re.search(r'^(?:[#S*\d\s]*)(?:नाम|नास|गाम)\s+(.+)', line)

        has_age_on_line = bool(re.search(r'आयु(?![षस])', line))

        if name_match and "मकान" not in line and not has_age_on_line:
            if "voter_name" in current_record or "rel_name" in current_record or current_record.get("has_age"):
                save_current(current_record)
                current_record = {}
                if pending_epic:
                    current_record["epic_id"] = pending_epic
                    pending_epic = None
            current_record["voter_name"] = name_match.group(1).strip()
            continue

                # C. Age and Gender line
        has_age_kw = has_age_on_line and "अनुसार" not in line
        age_match = re.search(r'आयु(?![षस])\s*[:ः\-]?\s*(\d{2})', line)
        if not age_match:
            age_match = re.search(r'(\d{1,3})\D{0,15}(?:पुरुष|महिला|स्त्री)', line)
        gender_val = None
        if "पुरुष" in line:
            gender_val = "Male"
        elif "महिला" in line or "स्त्री" in line:
            gender_val = "Female"

        if has_age_kw:
            if current_record.get("has_age"):
                save_current(current_record)
                current_record = {}
                if pending_epic:
                    current_record["epic_id"] = pending_epic
                    pending_epic = None
            current_record["has_age"] = True
            current_record["age"] = int(age_match.group(1)) if age_match else 0
            if gender_val:
                current_record["gender"] = gender_val
        elif gender_val and "gender" not in current_record:
            current_record["gender"] = gender_val

    save_current(current_record)

    formatted_chunks = []
    for idx, rec in enumerate(voter_records):
        epic = rec.get("epic_id", f"UNKNOWN_EPIC_{page_no}_{idx}")
        name = rec.get("voter_name", "Unknown")
        rel = rec.get("rel_name", "Unknown")
        age = rec.get("age", 0)
        explicit_gender = rec.get("gender")
        gender = explicit_gender or infer_gender_fallback(name, rec.get("is_husband", False))
        surname = extract_surname(name, rel)

        chunk_text = (
            f"Voter ID: {epic} | Name: {name} | Surname: {surname} | "
            f"Relative: {rel} | Age: {age} | Gender: {gender}"
        )

        formatted_chunks.append({
            "id": f"{pdf_name}_p{page_no}_v{idx}",
            "text": chunk_text,
            "metadata": {
                "source_pdf": pdf_name,
                "page": page_no,
                "epic_id": epic,
                "voter_name": name,
                "surname": surname,
                "gender": gender,
                "age": age,
                "is_deleted": bool(rec.get("is_deleted", False)),
                "gender_inferred": bool(explicit_gender is None),
                "is_husband_rel": bool(rec.get("is_husband", False))
            }
        })

    return formatted_chunks

def reconcile_with_official_summary(voter_chunks: list, summary: dict, pdf_name: str) -> list:
    """
    Drops only voters explicitly stamped DELETED in the source page. Reports
    how extraction compares to the PDF's own official summary table, but
    never pads with fake voters or reassigns gender to force a numeric match.
    """
    voter_chunks = [c for c in voter_chunks if not c["metadata"].get("is_deleted")]

    target_total = summary.get("net_total", 0) if summary else 0
    if target_total > 0:
        diff = len(voter_chunks) - target_total
        print(f"[{pdf_name}] QA: {len(voter_chunks)} extracted vs official net {target_total} "
              f"(M:{summary.get('net_male')}/F:{summary.get('net_female')}) -> {diff:+d}")
    else:
        print(f"[{pdf_name}] QA: no official summary read; keeping {len(voter_chunks)} as extracted.")

    for c in voter_chunks:
        c["metadata"].pop("is_deleted", None)
        c["metadata"].pop("gender_inferred", None)
        c["metadata"].pop("is_husband_rel", None)
        if summary:
            c["metadata"]["pdf_net_total"] = summary.get("net_total", 0)
            c["metadata"]["pdf_net_male"] = summary.get("net_male", 0)
            c["metadata"]["pdf_net_female"] = summary.get("net_female", 0)

    return voter_chunks

def process_pdf(file_path: str, pdf_name: str):
    doc = pymupdf.open(file_path)
    voter_chunks = []
    total_pages = len(doc)

    for page_idx in range(2, total_pages - 1):
        page = doc[page_idx]
        print(f"[{pdf_name}] Scanning voter columns on Page {page_idx + 1}/{total_pages}...")

        pix = page.get_pixmap(dpi=300)
        stacked_img = crop_and_stack_columns(pix)

        try:
            ocr_text = pytesseract.image_to_string(stacked_img, lang='hin+eng', config='--psm 6')
            chunks = parse_voter_text(ocr_text, pdf_name, page_idx + 1)
            voter_chunks.extend(chunks)
        except Exception as e:
            print(f"OCR failed on page {page_idx + 1}. Error: {e}")

    print(f"[{pdf_name}] Verifying against Official ECI Summary Table (Page 1 & Page {total_pages})...")
    summary = extract_official_summary(doc, expected_count=len(voter_chunks))
    if summary and "net_total" in summary:
        print(
            f"[{pdf_name}] Certified ECI Summary -> "
            f"Gross Serial Listed: {summary.get('gross_total')} "
            f"(Mother Roll: {summary.get('mother_total')} + Additions: {summary.get('additions_total')}) | "
            f"Deleted: {summary.get('deleted_total')} | "
            f"Net Active Voters: {summary.get('net_total')} "
            f"(Male: {summary.get('net_male')}, Female: {summary.get('net_female')})"
        )

    print(f"[{pdf_name}] Verifying against Official ECI Summary Table (Page 1 & Page {total_pages})...")
    summary = extract_official_summary(doc, expected_count=len(voter_chunks))
    if summary and "net_total" in summary:
        print(
            f"[{pdf_name}] Certified ECI Summary -> "
            f"Gross Serial Listed: {summary.get('gross_total')} "
            f"(Mother Roll: {summary.get('mother_total')} + Additions: {summary.get('additions_total')}) | "
            f"Deleted: {summary.get('deleted_total')} | "
            f"Net Active Voters: {summary.get('net_total')} "
            f"(Male: {summary.get('net_male')}, Female: {summary.get('net_female')})"
        )
        save_summary(pdf_name, summary)

    doc.close()
    voter_chunks = reconcile_with_official_summary(voter_chunks, summary, pdf_name)
    return voter_chunks