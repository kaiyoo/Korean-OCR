# Project Setup & Training Guide

## Environment Setup

Python 3.8.18

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

---

## Directory Structure

학습 전에 아래 스크립트 실행시 생성되는 폴더 구조:

```
Project/
├── model/
│   └── korean_PP-OCRv3_rec_train/
│        ├── best_accuracy.pdopt
│        ├── best_accuracy.pdparams
│        ├── best_accuracy.states
│        ├── config.yml
│        └── customized_korean_dict.txt
├── data/
│   └── processed/
│        ├── train/
│        │    ├── images/
│        │    └── train_label.txt
│        └── val/
│             ├── images/
│             └── val_label.txt
```

---

## Training Workflow

### 1. 원고지 격자 합성 데이터 만들기
```bash
python synthesize_data.py
```
- 생성된 데이터:  
  `data/processed/train/images/`,  
  `data/processed/train/train_label.txt`

---

### 1.1. Numpy Cache 생성 (학습 실행 전 미리 데이터 캐시 생성 후 학습 시 빠른 데이터 로드)
```bash
python generate_cache.py -c ./model/korean_PP-OCRv3_rec_train/config.yml
```

---

### 2. Custom 한글 사전 만들기
```bash
python create_customized_dict.py
```
- 기존 3687자 → 데이터셋 내 글자만 사용해서 802자로 축소  
  생성된 사전:  
  `model/korean_PP-OCRv3_rec_train/customized_korean_dict.txt`

---

### 3. Config 수정
- 데이터, pretrained model 경로, image shape, batch size 등
→ config 파일 수정:  
`model/korean_PP-OCRv3_rec_train/config.yml`

---

### 4. 학습 실행
```bash
python tools/train.py -c ./model/korean_PP-OCRv3_rec_train/config.yml
```

---

## 코드 수정 내역

- **train.py**  
  → backbone freeze (현재 사용 안함)
- **tools/program.py**  
  → eval 시 prediction 샘플 출력이 나오도록 수정
- **ppocr/utils/logging.py**  
  → 터미널 출력 외에 `output//rec_ppocr_v3_korean/train.log` 파일에도 로그 저장되도록 수정
- **CachedSimpleDataSet 추가**  
  → 첫 로딩 시 npy cache 생성, 이후에는 캐시에서 빠른 로드  
    (위치: `ppocr/data/__init__.py`, `ppocr/data/simple_dataset.py`)

---
