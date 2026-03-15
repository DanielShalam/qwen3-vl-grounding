"""Run grouped single-class inference with temperature ablation.

Uses lvis_validation_grouped.json. For each sample, asks the model to locate
all instances of one category. Parses all predicted boxes+labels from response.
Evaluates using label-aware best-IoU matching against grouped GT.
"""
import json, re, argparse, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def parse_all_bboxes(response):
    """Parse all bbox_2d + label pairs from model response."""
    results = []
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response):
        box = [int(m.group(i)) for i in range(1, 5)]
        # Find label near this bbox
        start = max(0, m.start() - 50)
        end = min(len(response), m.end() + 50)
        ctx = response[start:end]
        lm = re.search(r'"label"\s*:\s*"([^"]+)"', ctx)
        label = lm.group(1).lower() if lm else None
        results.append({"box": box, "label": label})
    return results


def calculate_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def run(model_name, limit, temperatures, output_base):
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Load grouped validation + build GT lookup
    grouped = json.load(open("/efs/user_folders/dnshalam/datasets/lvis/lvis_validation_grouped.json"))
    gt_lookup = {}
    for g in grouped:
        boxes = json.loads(g["conversations"][1]["value"])
        gt_lookup[(g["id"], g["category"])] = [b["bbox_2d"] for b in boxes]

    # Use first `limit` samples from grouped validation
    samples = grouped[:limit]
    print(f"Running on {len(samples)} samples")

    for temp in temperatures:
        print(f"\n=== Temperature: {temp} ===")
        predictions = []
        gen_kwargs = {"max_new_tokens": 256}
        if temp == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update({"do_sample": True, "temperature": temp, "top_p": 0.95})

        for item in tqdm(samples, desc=f"T={temp}"):
            image_path = item["image"]
            asked_cat = item["category"]
            prompt = item["conversations"][0]["value"].replace("<image>\n", "")

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            response = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # Parse predicted boxes
            pred_bboxes = parse_all_bboxes(response)

            # Get GT boxes for this (image, category)
            gt_boxes = gt_lookup.get((item["id"], asked_cat), [])

            predictions.append({
                "id": item["id"],
                "asked_category": asked_cat,
                "gt_boxes": gt_boxes,
                "pred_bboxes": pred_bboxes,
                "response": response,
            })

        # Evaluate: label-aware best-IoU
        ious, correct, total, failed = [], 0, 0, 0
        for p in predictions:
            asked = p["asked_category"]
            gt_boxes = p["gt_boxes"]
            for pb in p["pred_bboxes"]:
                label_match = pb["label"] == asked if pb["label"] else False
                # IoU against GT boxes of asked category
                if gt_boxes:
                    best = max(calculate_iou(gb, pb["box"]) for gb in gt_boxes)
                else:
                    best = 0.0
                ious.append(best)
                total += 1
                if best >= 0.5:
                    correct += 1
            if not p["pred_bboxes"]:
                failed += 1

        mean_iou = sum(ious) / len(ious) if ious else 0
        acc = correct / total if total else 0
        label_matches = sum(1 for p in predictions for pb in p["pred_bboxes"]
                           if pb["label"] == p["asked_category"])

        print(f"  Evaluated: {total}, Failed parse: {failed}")
        print(f"  Label match: {label_matches}/{total}")
        print(f"  Mean IoU: {mean_iou:.4f}")
        print(f"  Accuracy@0.5: {acc:.4f} ({correct}/{total})")

        # Save
        out_dir = Path(output_base) / f"temp_{temp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump({"temperature": temp, "total": total, "failed": failed,
                        "label_matches": label_matches, "mean_iou": mean_iou,
                        "accuracy": acc, "correct": correct, "threshold": 0.5}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--output_dir", default="results/ablation_temperature")
    args = parser.parse_args()
    run(args.model, args.limit, args.temperatures, args.output_dir)
