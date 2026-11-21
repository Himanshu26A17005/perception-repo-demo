import os
import csv
import glob
import yaml
import cv2
import numpy as np

# Runs directly inside your unzipped dataset directory
DATASET_ROOT = "."
DIGITS_DIR = "extracted_digits"        # Single digits 0..9 for training recognizers
BENCHMARK_DIR = "benchmark_0_to_11"    # Test stage signs numbered 0 to 11

for d in range(10):
    os.makedirs(f"{DIGITS_DIR}/{d}", exist_ok=True)
os.makedirs(f"{BENCHMARK_DIR}/images", exist_ok=True)
os.makedirs(f"{BENCHMARK_DIR}/labels", exist_ok=True)

# Exact mapping for your 13 classes: '10 MPH' -> "10", '15 MPH' -> "15", etc.
CLASS_TO_NUM = {
    0: "10", 1: "15", 2: "20", 3: "25", 4: "30", 5: "35", 6: "40",
    7: "45", 8: "50", 9: "55", 10: "60", 11: "65", 12: "70"
}


def segment_two_digits(sign_crop: np.ndarray):
    """Extracts the 2 speed-limit digits (ignores the smaller 'SPEED LIMIT' & 'MPH' text)."""
    h, w = sign_crop.shape[:2]
    # US MPH signs have 'SPEED LIMIT' at top ~28% and 'MPH' at bottom ~22%
    # The big two-digit number sits in the middle vertical band (25% to 78% height)
    roi = sign_crop[int(h * 0.25):int(h * 0.78), int(w * 0.12):int(w * 0.88)]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 4)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rh, rw = th.shape
    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Filter for the two large central digits
        if (rh * 0.38) < ch < (rh * 0.96) and (rw * 0.06) < cw < (rw * 0.55):
            boxes.append((x, y, cw, ch))

    # Keep the 2 largest contours by height if extra noise contours exist
    boxes = sorted(boxes, key=lambda b: b[3], reverse=True)[:2]
    if len(boxes) != 2:
        return []

    # Sort strictly left-to-right (tens digit first, ones digit second)
    boxes = sorted(boxes, key=lambda b: b[0])

    crops = []
    for (x, y, cw, ch) in boxes:
        d_roi = th[y:y + ch, x:x + cw]
        side = max(cw, ch) + 6
        canvas = np.zeros((side, side), dtype=np.uint8)
        ox, oy = (side - cw) // 2, (side - ch) // 2
        canvas[oy:oy + ch, ox:ox + cw] = d_roi
        crops.append(cv2.resize(canvas, (28, 28)))
    return crops


def estimate_lighting_and_angle(img: np.ndarray, bbox: list):
    """Classifies lighting and viewing angle from sign crop."""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mean_v, std_v = float(np.mean(v)), float(np.std(v))
    glare_ratio = float(np.sum(v > 245)) / float(v.size)

    if mean_v < 85:
        lighting = "low_light"
    elif glare_ratio > 0.10:
        lighting = "glare"
    elif std_v > 62:
        lighting = "shadow"
    else:
        lighting = "normal"

    # US MPH signs have a standard 4:5 (0.8) width-to-height aspect ratio
    w, h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
    observed_ratio = (w / h) / 0.80
    est_angle = float(np.rad2deg(np.arccos(np.clip(observed_ratio, 0.0, 1.0))))

    if est_angle < 20:
        angle = "frontal_0_20"
    elif est_angle < 40:
        angle = "moderate_20_40"
    else:
        angle = "extreme_40_plus"

    return lighting, angle, round(est_angle, 1)


def populate_8_and_9_plus_augment(digit_counts: dict, min_samples: int = 60):
    """
    1. Rotates extracted real '6's by 180 deg to create real-world '9's.
    2. Adds synthetic font samples so all digits 0-9 have >= min_samples.
    """
    six_files = glob.glob(f"{DIGITS_DIR}/6/*.png")
    for s_path in six_files:
        img6 = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)
        if img6 is not None:
            img9 = cv2.rotate(img6, cv2.ROTATE_180)
            idx = digit_counts["9"]
            cv2.imwrite(f"{DIGITS_DIR}/9/rot6_{idx:04d}.png", img9)
            digit_counts["9"] += 1

    fonts = [cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
    for d in range(10):
        d_str = str(d)
        while digit_counts[d_str] < min_samples:
            idx = digit_counts[d_str]
            canvas = np.zeros((28, 28), dtype=np.uint8)
            font = fonts[idx % len(fonts)]
            scale = np.random.uniform(0.68, 0.86)
            thick = np.random.choice([1, 2, 2])
            (tw, th), _ = cv2.getTextSize(d_str, font, scale, thick)
            tx = (28 - tw) // 2 + np.random.randint(-2, 3)
            ty = (28 + th) // 2 + np.random.randint(-2, 3)
            cv2.putText(canvas, d_str, (tx, ty), font, scale, 255, thick, cv2.LINE_AA)
            cv2.imwrite(f"{DIGITS_DIR}/{d_str}/synth_{idx:04d}.png", canvas)
            digit_counts[d_str] += 1


def main():
    digit_counts = {str(d): 0 for d in range(10)}
    manifest_rows = []
    test_idx = 0

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        lbl_dir = os.path.join(DATASET_ROOT, split, "labels")
        if not os.path.exists(img_dir):
            continue

        for lbl_path in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(lbl_path))[0]
            img_matches = glob.glob(os.path.join(img_dir, stem + ".*"))
            if not img_matches:
                continue
            img = cv2.imread(img_matches[0])
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(lbl_path, "r") as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]
            if not lines:
                continue

            parts = lines[0].split()
            cid = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            x1, y1 = max(0, int((xc - bw / 2) * w)), max(0, int((yc - bh / 2) * h))
            x2, y2 = min(w - 1, int((xc + bw / 2) * w)), min(h - 1, int((yc + bh / 2) * h))
            box_w, box_h = x2 - x1, y2 - y1
            if box_w < 28 or box_h < 28:
                continue

            sign_crop = img[y1:y2, x1:x2]

            # 1. Slice real digits (0..7) from the MPH sign
            if cid in CLASS_TO_NUM:
                num_str = CLASS_TO_NUM[cid]
                crops = segment_two_digits(sign_crop)
                for ch_d, d_img in zip(num_str, crops):
                    cv2.imwrite(f"{DIGITS_DIR}/{ch_d}/{stem}_{digit_counts[ch_d]:04d}.png", d_img)
                    digit_counts[ch_d] += 1

            # 2. Repaint the central speed number into Stage 0..11 while keeping real sign + lighting
            stage_num = test_idx % 12  # Cycles 0, 1, 2, ..., 10, 11
            ix1, iy1 = x1 + int(box_w * 0.14), y1 + int(box_h * 0.24)
            ix2, iy2 = x2 - int(box_w * 0.14), y2 - int(box_h * 0.20)
            patch = img[iy1:iy2, ix1:ix2]
            if patch.size == 0:
                continue

            # Match the local white board color and dark ink intensity of that exact photo
            bg_col = np.percentile(patch, 85, axis=(0, 1)).astype(np.uint8).tolist()
            fg_col = np.percentile(patch, 10, axis=(0, 1)).astype(np.uint8).tolist()
            cv2.rectangle(img, (ix1, iy1), (ix2, iy2), bg_col, -1)

            text = str(stage_num)
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = (min(box_w, box_h) / 58.0) * (0.80 if stage_num >= 10 else 0.95)
            thick = max(1, int(scale * 2.2))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            tx = ix1 + ((ix2 - ix1) - tw) // 2
            ty = iy1 + ((iy2 - iy1) + th) // 2
            cv2.putText(img, text, (tx, ty), font, scale, fg_col, thick, cv2.LINE_AA)

            lighting, angle_bkt, est_deg = estimate_lighting_and_angle(img, [x1, y1, x2, y2])
            out_fname = f"stage_{stage_num:02d}_{test_idx:04d}.jpg"
            cv2.imwrite(f"{BENCHMARK_DIR}/images/{out_fname}", img)

            manifest_rows.append([
                out_fname, stage_num, lighting, angle_bkt, est_deg, x1, y1, x2, y2
            ])
            test_idx += 1

    populate_8_and_9_plus_augment(digit_counts)

    manifest_csv = f"{BENCHMARK_DIR}/benchmark_manifest.csv"
    with open(manifest_csv, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["filename", "gt_number", "lighting", "angle_bucket", "est_deg", "x1", "y1", "x2", "y2"])
        writer.writerows(manifest_rows)

    print(f"Created {test_idx} Stage 0-11 test signs in '{BENCHMARK_DIR}/images'")
    print("Digit sample counts (0-9):", digit_counts)


if __name__ == "__main__":
    main()