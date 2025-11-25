import json
import time
import argparse
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, DistilBertTokenizerFast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--max_length", type=int, default=128) # Updated to match training
    ap.add_argument("--runs", type=int, default=100) # Increase runs for stability
    ap.add_argument("--device", default="cpu") # Force CPU as per requirement
    args = ap.parse_args()

    model_path = args.model_dir if args.model_name is None else args.model_name
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    # Quantize if on CPU for speed
    if args.device == "cpu":
        # Set engine for ARM64 (Apple Silicon) or x86
        import platform
        if platform.machine() == "arm64":
            torch.backends.quantized.engine = 'qnnpack'
        else:
            torch.backends.quantized.engine = 'fbgemm'
            
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
    model.to(args.device)
    model.eval()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    if not texts:
        print("No texts found in input file.")
        return

    times_ms = []

    # warmup
    print("Warming up...")
    for _ in range(10):
        t = texts[0]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))

    print(f"Running {args.runs} inferences...")
    for i in range(args.runs):
        t = texts[i % len(texts)]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]

    print(f"Latency over {args.runs} runs (batch_size=1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")

if __name__ == "__main__":
    main()
