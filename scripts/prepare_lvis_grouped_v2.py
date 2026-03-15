"""Prepare grouped LVIS data for Qwen3-VL finetuning.

Changes from v1:
1. Adds "label" field to each bbox (matches Qwen3-VL native output format)
2. Sorts boxes by (y1, x1) reading order for consistent training
"""
import json
import argparse
from pathlib import Path


def prepare(input_path, output_path):
    data = json.load(open(input_path))
    out = []
    for item in data:
        boxes = json.loads(item["conversations"][1]["value"])
        # Add label and sort by (y1, x1)
        labeled = [{"bbox_2d": b["bbox_2d"], "label": item["category"]} for b in boxes]
        labeled.sort(key=lambda x: (x["bbox_2d"][1], x["bbox_2d"][0]))
        new_item = {
            "id": item["id"],
            "image": item["image"],
            "conversations": [
                item["conversations"][0],
                {"from": "gpt", "value": json.dumps(labeled)}
            ]
        }
        out.append(new_item)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.input, args.output)
