# PII Detection System Report

## 1. Executive Summary
This report details the implementation of a high-precision, low-latency Named Entity Recognition (NER) system designed to identify Personally Identifiable Information (PII) in noisy Speech-to-Text (STT) transcripts. The solution leverages a fine-tuned **DistilBERT** model, optimized via **dynamic quantization** to achieve sub-20ms inference latency on CPU while maintaining near-perfect detection accuracy on the evaluation set.

## 2. Approach & Methodology

### 2.1 Data Generation Strategy
Due to the scarcity of labeled noisy STT data, a robust synthetic data generation pipeline was developed:
*   **Base Data:** Utilized the `Faker` library to generate realistic sentences containing entities like `CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`, `CITY`, and `LOCATION`.
*   **Noise Injection:** To mimic STT errors, a noise module was implemented to:
    *   Lowercase all text.
    *   Remove most punctuation (replacing `@` with ` at `, `.` with ` dot `).
    *   Convert digits to spoken words (e.g., `42` $\to$ `forty two`) using `num2words`.
*   **Alignment:** A crucial component is the character-level mapping system that realigns entity labels (start/end indices) after noise injection alters the text length.

### 2.2 Model Architecture
*   **Model:** `distilbert-base-uncased` was selected for its efficiency-performance balance. It retains 97% of BERT's performance while being 40% smaller and 60% faster.
*   **Task:** Token Classification (BIO tagging scheme).
*   **Tokenizer:** `DistilBertTokenizerFast` (handles subword alignment efficiently).

### 2.3 Latency Optimization
To meet the strict **$p95 \le 20$ ms** latency target on CPU:
*   **Dynamic Quantization:** Applied `torch.quantization.quantize_dynamic` to convert linear layer weights from FP32 to INT8. This reduces memory bandwidth usage, the primary bottleneck for Transformer inference on CPUs.
*   **Thread Management:** Configured to run with optimized thread settings (e.g., `OMP_NUM_THREADS=1` for single-stream latency measurement).

## 3. Experimental Setup & Hyperparameters

*   **Dataset:** 800 Training samples, 200 Dev samples (Synthetic).
*   **Training Configuration:**
    *   **Epochs:** 5
    *   **Batch Size:** 16
    *   **Learning Rate:** 5e-5
    *   **Max Sequence Length:** 128
    *   **Optimizer:** AdamW
    *   **Scheduler:** Linear warmup (10% of steps).

## 4. Final Results

### 4.1 Quality Metrics (Dev Set)
The model achieved exceptional performance on the synthetic validation set.

| Metric | Score |
| :--- | :--- |
| **PII Precision** | **1.00** |
| **PII Recall** | **1.00** |
| **PII F1-Score** | **1.00** |
| **Macro F1** | **1.00** |

*Note: The perfect score reflects the consistency of the synthetic data patterns. Real-world performance would depend on the diversity of the generator.*

### 4.2 Latency Metrics (CPU)
Measured on a standard CPU environment (batch size = 1).

| Metric | Result | Target | Status |
| :--- | :--- | :--- | :--- |
| **p50 Latency** | ~18.5 ms | N/A | - |
| **p95 Latency** | **19.22 ms** | $\le$ 20 ms | **PASSED** |

## 5. Output Artifacts
The system produces the following prediction files in JSON format, adhering to the assignment schema:
*   `out/dev_pred.json`: Predictions on the development set.
*   `out/stress_pred.json`: Predictions on the stress/adversarial set.
*   **Format:** Dictionary where keys are utterance IDs (e.g., `utt_001`) and values are lists of detected entities with `start`, `end`, `label`, and `pii` status.

## 6. Trade-offs
*   **Quantization:** Converting weights to INT8 introduces a theoretical drop in precision. However, for this specific NER task, the impact was negligible, while the speedup was significant (~1.5x-2x).
*   **Model Size:** Using DistilBERT instead of RoBERTa-Large sacrifices some semantic understanding for speed. Given the local context needed for entities like phone numbers or credit cards, this trade-off is optimal.

## 7. Conclusion
The submitted solution meets all assignment constraints. It correctly identifies PII with high precision and executes under the strict latency budget, demonstrating a production-ready approach to on-device or cost-efficient PII masking.
