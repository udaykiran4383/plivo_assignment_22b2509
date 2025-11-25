import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, DistilBertTokenizerFast
from labels import ID2LABEL, label_is_pii
import os
from tqdm import tqdm

def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # Skip special tokens (offset 0,0 usually, or if start==end)
        if start == end:
            continue
            
        # Also skip if start/end are out of bounds of text (shouldn't happen with correct offsets)
        if start >= len(text):
            continue

        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--text", default=None, help="Input text for single inference")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    results = {}

    if args.text:
        lines = [json.dumps({"id": "cli_input", "text": args.text})]
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = f.readlines()

    for line in tqdm(lines, desc="Predicting"):
        obj = json.loads(line)
        text = obj["text"]
        uid = obj["id"]

        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0]
            pred_ids = logits.argmax(dim=-1).cpu().tolist()

        spans = bio_to_spans(text, offsets, pred_ids)
        ents = []
        for s, e, lab in spans:
            ents.append(
                {
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                }
            )
        results[uid] = ents

    if args.text:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
