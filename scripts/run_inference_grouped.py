"""Run inference on grouped LVIS data (multi-instance grounding)."""
import json
import re
import argparse
import torch
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

DATA_DIR = Path("/efs/user_folders/dnshalam/datasets/lvis")


def parse_all_bboxes(response):
    return [[int(m.group(i)) for i in range(1, 5)]
            for m in re.finditer(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response)]


def run_inference(mode="baseline", config_path="configs/lora_config.yaml", limit=None,
                  adapter_path="checkpoints/final", output_dir=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    print(f"Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    if mode == "finetuned":
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    with open(DATA_DIR / "lvis_validation_grouped.json") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    predictions = []
    print(f"Running {mode} inference on {len(data)} grouped samples...")

    for item in tqdm(data):
        image_path = item["image"]
        prompt_text = item["conversations"][0]["value"].replace("<image>\n", "")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

        # Parse all GT boxes from the answer
        gt_boxes = parse_all_bboxes(item["conversations"][1]["value"])
        pred_boxes = parse_all_bboxes(response)

        predictions.append({
            "id": item["id"],
            "category": item["category"],
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes),
            "ground_truth_boxes": gt_boxes,
            "predicted_boxes": pred_boxes,
            "response": response,
        })

    out_dir = Path(output_dir) if output_dir else Path(f"results/{mode}_grouped")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "predictions.json"

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetuned"], default="baseline")
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adapter", default="checkpoints/final")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    run_inference(args.mode, args.config, args.limit, args.adapter, args.output_dir)
