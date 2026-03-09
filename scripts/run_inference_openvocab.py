"""Run inference with open prompt: detect all objects without specifying categories."""
import json
import re
import argparse
import torch
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

DATA_DIR = Path("/efs/user_folders/dnshalam/datasets/lvis")

PROMPT = "Locate all objects in this image and output the bbox coordinates in JSON format."


def parse_all_bboxes(response):
    boxes = []
    for m in re.finditer(r'\{[^}]*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\][^}]*"label"\s*:\s*"([^"]+)"[^}]*\}', response):
        boxes.append({"bbox_2d": [int(m.group(i)) for i in range(1, 5)], "label": m.group(5)})
    if boxes:
        return boxes
    for m in re.finditer(r'\{[^}]*"label"\s*:\s*"([^"]+)"[^}]*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\][^}]*\}', response):
        boxes.append({"bbox_2d": [int(m.group(i)) for i in range(2, 6)], "label": m.group(1)})
    return boxes


def run_inference(config_path="configs/lora_config.yaml", limit=None, output_dir=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    print(f"Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    with open(DATA_DIR / "lvis_validation_multiclass.json") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    predictions = []
    print(f"Running open-vocab inference on {len(data)} images...")

    for item in tqdm(data):
        try:
            image = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            print(f"Skipping {item['image']}: {e}")
            continue

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=4096)

        response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

        gt_boxes = parse_all_bboxes(item["conversations"][1]["value"])
        pred_boxes = parse_all_bboxes(response)

        predictions.append({
            "id": item["id"],
            "image": item["image"],
            "categories": item["categories"],
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes),
            "ground_truth_boxes": [b["bbox_2d"] for b in gt_boxes],
            "ground_truth_labels": [b["label"] for b in gt_boxes],
            "predicted_boxes": [b["bbox_2d"] for b in pred_boxes],
            "predicted_labels": [b["label"] for b in pred_boxes],
            "response": response,
        })

    out_dir = Path(output_dir) if output_dir else Path("results/baseline_openvocab")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "predictions.json"

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    run_inference(args.config, args.limit, args.output_dir)
