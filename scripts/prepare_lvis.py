import json
from pathlib import Path
from tqdm import tqdm

COCO_DIR = Path("/efs/user_folders/dnshalam/datasets/coco2017")
LVIS_DIR = Path("/efs/user_folders/dnshalam/datasets/lvis")
OUTPUT_DIR = LVIS_DIR

SPLIT_CONFIG = {
    "validation": {"annotations": LVIS_DIR / "lvis_v1_val.json", "images_dir": COCO_DIR / "val2017"},
    "train": {"annotations": LVIS_DIR / "lvis_v1_train.json", "images_dir": COCO_DIR / "train2017"},
}


def format_bbox_to_qwen(bbox, img_width, img_height):
    """Convert LVIS bbox [x, y, width, height] to Qwen normalized format"""
    x, y, w, h = bbox
    x1_norm = int((x / img_width) * 1000)
    y1_norm = int((y / img_height) * 1000)
    x2_norm = int(((x + w) / img_width) * 1000)
    y2_norm = int(((y + h) / img_height) * 1000)
    return f"<box>({x1_norm},{y1_norm}),({x2_norm},{y2_norm})</box>"


def process_split(split_name):
    output_file = OUTPUT_DIR / f"lvis_{split_name}.json"
    if output_file.exists():
        print(f"Skipping {split_name} - already exists at {output_file}")
        return

    config = SPLIT_CONFIG[split_name]
    print(f"\nLoading LVIS {split_name} annotations...")
    with open(config["annotations"]) as f:
        lvis_data = json.load(f)

    # Build lookup maps
    images = {img["id"]: img for img in lvis_data["images"]}
    categories = {cat["id"]: cat["name"] for cat in lvis_data["categories"]}

    conversations = []
    print(f"Processing {len(lvis_data['annotations'])} annotations...")
    for ann in tqdm(lvis_data["annotations"]):
        img = images[ann["image_id"]]
        img_w, img_h = img["width"], img["height"]
        filename = img["coco_url"].split("/")[-1]
        image_path = str(config["images_dir"] / filename)
        category = categories[ann["category_id"]]

        bbox_str = format_bbox_to_qwen(ann["bbox"], img_w, img_h)
        conversations.append({
            "id": ann["image_id"],
            "image": image_path,
            "conversations": [
                {"from": "human", "value": f"<img>{image_path}</img>\nLocate the {category} in this image."},
                {"from": "gpt", "value": f"The {category} is located at {bbox_str}"},
            ],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(conversations, f, indent=2)

    print(f"Saved {len(conversations)} conversations to {output_file}")


def main():
    process_split("validation")
    process_split("train")


if __name__ == "__main__":
    main()
