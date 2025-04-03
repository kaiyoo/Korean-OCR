import os

# ====== 경로 설정 ======
train_label_path = "./data/processed/train/train_label.txt"
val_label_path = "./data/processed/val/val_label.txt"
output_dict_path = "./model/customized_korean_dict.txt"

# ====== 모든 라벨 텍스트 읽기 ======
all_text = ""
for path in [train_label_path, val_label_path]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                all_text += parts[1]

# ====== Unique characters 추출 ======
unique_chars = sorted(set(all_text))

print(f"✅ 총 {len(unique_chars)}개의 unique character가 추출되었습니다.")

# ====== 파일로 저장 ======
with open(output_dict_path, "w", encoding="utf-8") as f:
    for ch in unique_chars:
        f.write(ch + "\n")

print(f"🎯 저장 완료 → {output_dict_path}")
