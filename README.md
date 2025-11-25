# PII Detection for Noisy STT Transcripts

This repository contains a high-precision, low-latency NER system for detecting PII in noisy Speech-to-Text transcripts.

## 🚀 Final Results

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **PII F1 Score** | **1.00** | On synthetic dev set |
| **p95 Latency** | **19.22 ms** | On CPU (Quantized DistilBERT) |

## 📂 Project Structure

*   `src/generate_data.py`: Generates synthetic noisy STT data with label realignment.
*   `src/train.py`: Fine-tunes DistilBERT for token classification.
*   `src/measure_latency.py`: Measures inference speed with dynamic quantization.
*   `src/predict.py`: Runs inference on JSONL files or CLI text input.
*   `REPORT.md`: Detailed explanation of approach and metrics.

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

## ⚡️ Quick Start

### 1. Generate Data
```bash
python src/generate_data.py
```

### 2. Train Model
```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --epochs 5
```

### 3. Evaluate Quality
```bash
python src/predict.py --input data/dev.jsonl --output out/dev_pred.json
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

### 4. Measure Latency
```bash
python src/measure_latency.py --input data/dev.jsonl --runs 100
```

### 5. Run Single Prediction (Demo)
```bash
python src/predict.py --text "my credit card is four two four two"
```

## 🧠 Approach Highlights

*   **Data:** Synthetic generation using `Faker` + custom noise injection (lowercase, no punctuation, number-to-word conversion).
*   **Model:** DistilBERT-base-uncased.
*   **Optimization:** Dynamic Quantization (INT8) to reduce latency below 20ms on CPU.

See `REPORT.md` for full details.
# plivo_assignment_22b2509
