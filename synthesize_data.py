import os
import cv2
import numpy as np
import math
from tqdm import tqdm

# ====== 설정 ======
data_root = "./data"
output_root = "./data/processed"

sets = [("train_img", "train_txt", "train"), ("val_img", "val_txt", "val")]
cell_ratio = 0.9
line_thickness = 2  # 선 굵기

os.makedirs(output_root, exist_ok=True)

def apply_grid(image_path, save_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    cell_size = int(h * cell_ratio)
    cols = math.ceil(w / cell_size)
    rows = math.ceil(h / cell_size)

    # Grid canvas
    grid_canvas = np.ones_like(image) * 255

    # Grid lines
    for row in range(rows + 1):
        y = row * cell_size
        cv2.line(grid_canvas, (0, y), (w, y), (0, 0, 0), line_thickness)
    for col in range(cols + 1):
        x = col * cell_size
        cv2.line(grid_canvas, (x, 0), (x, h), (0, 0, 0), line_thickness)
    cv2.rectangle(grid_canvas, (0, 0), (w - 1, h - 1), (0, 0, 0), line_thickness)

    # Blend
    result = image.copy()
    mask = (grid_canvas < 128)
    result[mask] = grid_canvas[mask]

    # Save
    cv2.imwrite(save_path, result)
    return cell_size, rows, cols

for img_folder, txt_folder, set_name in sets:
    output_img_dir = os.path.join(output_root, set_name, "images")
    os.makedirs(output_img_dir, exist_ok=True)
    label_output_path = os.path.join(output_root, set_name, f"{set_name}_label.txt")

    img_list = sorted(os.listdir(os.path.join(data_root, img_folder)))
    txt_list = sorted(os.listdir(os.path.join(data_root, txt_folder)))

    assert len(img_list) == len(txt_list), f"{set_name} 데이터 수 불일치!"

    label_lines = []

    print(f"🔄 {set_name.upper()} 데이터 변환 중... ({len(img_list)}개)")
    for img_name, txt_name in tqdm(zip(img_list, txt_list), total=len(img_list)):
        img_path = os.path.join(data_root, img_folder, img_name)
        txt_path = os.path.join(data_root, txt_folder, txt_name)
        output_img_path = os.path.join(output_img_dir, img_name)

        # Apply Grid
        apply_grid(img_path, output_img_path)

        # Read label
        with open(txt_path, "r", encoding="utf-8") as f:
            label_text = f.read().strip()

        # Add to label file
        rel_img_path = os.path.join("images", img_name)
        label_lines.append(f"{rel_img_path}\t{label_text}")

    # Save label txt
    os.makedirs(os.path.join(output_root, set_name), exist_ok=True)
    with open(label_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))

    print(f"✅ {set_name} 처리 완료 → {output_img_dir}, {label_output_path}\n")
