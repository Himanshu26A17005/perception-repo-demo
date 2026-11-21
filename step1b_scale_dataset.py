import os
import csv
import cv2
import numpy as np

SRC_DIR = "benchmark_0_to_11"
DST_DIR = "benchmark_540"
os.makedirs(f"{DST_DIR}/images", exist_ok=True)

# Strictly held-out test fonts (never used in training)
UNSEEN_FONTS = [
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
]

def repaint_sign(img, box, num_str, font):
    out = img.copy()
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rx1, ry1 = x1 + int(w * 0.14), y1 + int(h * 0.24)
    rx2, ry2 = x1 + int(w * 0.86), y1 + int(h * 0.80)
    if rx2 <= rx1 or ry2 <= ry1:
        return out

    bg_patch = out[y1 + 2:ry1, rx1:rx2]
    bg_color = np.median(bg_patch.reshape(-1, 3), axis=0).astype(int).tolist() if bg_patch.size > 0 else [220, 220, 220]
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), bg_color, -1)

    rw, rh = rx2 - rx1, ry2 - ry1
    scale = max(0.35, min(rw, rh) / (34.0 if len(num_str) == 2 else 24.0))
    thick = max(1, int(round(scale * 2.0)))
    (tw, th), _ = cv2.getTextSize(num_str, font, scale, thick)
    tx = rx1 + max(1, (rw - tw) // 2)
    ty = ry1 + max(th + 1, (rh + th) // 2)

    cv2.putText(out, num_str, (tx, ty), font, scale, [25, 25, 25], thick, cv2.LINE_AA)
    return out

def apply_condition(img, box, mode):
    out = img.copy().astype(np.float32)
    x1, y1, x2, y2 = box

    if "glare" in mode:
        out[y1:y2, x1:x2] = np.clip(out[y1:y2, x1:x2] * 1.35 + 45, 0, 255)
    if "shadow" in mode:
        out[y1:y2, x1:x2] = np.clip(out[y1:y2, x1:x2] * 0.48, 0, 255)
    if "low_light" in mode:
        out = np.clip(out * 0.55 + np.random.normal(0, 8.0, out.shape), 0, 255)
    if "blur" in mode:
        kernel = np.zeros((5, 5), dtype=np.float32)
        kernel[2, :] = 0.2
        out = cv2.filter2D(out, -1, kernel)
    if "noise" in mode:
        out = np.clip(out + np.random.normal(0, 10.0, out.shape), 0, 255)

    return out.astype(np.uint8)

def main():
    np.random.seed(42)
    with open(f"{SRC_DIR}/benchmark_manifest.csv", "r") as f:
        rows = list(csv.DictReader(f))

    # 10 distinct conditions per base sign -> 540 total test signs
    variants = [
        ("normal", "normal"),
        ("normal", "blur"),
        ("normal", "noise"),
        ("glare", "glare"),
        ("glare", "glare_blur"),
        ("shadow", "shadow"),
        ("shadow", "shadow_blur"),
        ("low_light", "low_light"),
        ("low_light", "low_light_blur"),
        ("low_light", "low_light_noise"),
    ]

    new_rows = []
    idx = 0
    for r in rows:
        img = cv2.imread(f"{SRC_DIR}/images/{r['filename']}")
        if img is None:
            continue
        box = [int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])]

        for v_i, (light_tag, mode) in enumerate(variants):
            gt_num = (int(r["gt_number"]) + v_i) % 12
            font = UNSEEN_FONTS[idx % len(UNSEEN_FONTS)]
            aug = repaint_sign(img, box, str(gt_num), font)
            aug = apply_condition(aug, box, mode)

            fname = f"stage_{gt_num:02d}_{idx:04d}.jpg"
            cv2.imwrite(f"{DST_DIR}/images/{fname}", aug)
            new_rows.append({
                "filename": fname,
                "gt_number": gt_num,
                "lighting": light_tag,
                "angle_bucket": r["angle_bucket"],
                "est_deg": r["est_deg"],
                "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"]
            })
            idx += 1

    with open(f"{DST_DIR}/benchmark_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"Generated {len(new_rows)} diverse test signs in {DST_DIR}/")

if __name__ == "__main__":
    main()