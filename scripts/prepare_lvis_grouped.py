"""
Prepare LVIS dataset with multi-instance grouping.
Groups all annotations per image+category so the model learns to locate ALL instances.
"""
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

COCO_DIR = Path("/efs/user_folders/dnshalam/datasets/coco2017")
LVIS_DIR = Path("/efs/user_folders/dnshalam/datasets/lvis")
OUTPUT_DIR = LVIS_DIR

SPLIT_CONFIG = {
    "validation": {"annotations": LVIS_DIR / "lvis_v1_val.json", "images_dir": COCO_DIR / "val2017"},
    "train": {"annotations": LVIS_DIR / "lvis_v1_train.json", "images_dir": COCO_DIR / "train2017"},
}


def format_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    return [
        int((x / img_w) * 1000),
        int((y / img_h) * 1000),
        int(((x + w) / img_w) * 1000),
        int(((y + h) / img_h) * 1000),
    ]


def process_split(split_name):
    output_file = OUTPUT_DIR / f"lvis_{split_name}_grouped.json"
    config = SPLIT_CONFIG[split_name]

    print(f"\nLoading LVIS {split_name} annotations...")
    with open(config["annotations"]) as f:
        lvis_data = json.load(f)

    images = {img["id"]: img for img in lvis_data["images"]}
    categories = {cat["id"]: cat["name"] for cat in lvis_data["categories"]}

    # Group annotations by (image_id, category_id)
    groups = defaultdict(list)
    for ann in lvis_data["annotations"]:
        groups[(ann["image_id"], ann["category_id"])].append(ann["bbox"])

    conversations = []
    print(f"Processing {len(groups)} image+category groups...")
    for (img_id, cat_id), bboxes in tqdm(groups.items()):
        img = images[img_id]
        img_w, img_h = img["width"], img["height"]
        filename = img["coco_url"].split("/")[-1]
        image_path = str(config["images_dir"] / filename)
        category = categories[cat_id]

        formatted_bboxes = [{"bbox_2d": format_bbox(b, img_w, img_h)} for b in bboxes]
        answer = json.dumps(formatted_bboxes)

        conversations.append({
            "id": img_id,
            "image": image_path,
            "category": category,
            "num_instances": len(bboxes),
            "conversations": [
                {"from": "human", "value": f"<image>\nLocate all {category} in this image and output the bbox coordinates in JSON format."},
                {"from": "gpt", "value": answer},
            ],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(conversations, f, indent=2)

    single = sum(1 for c in conversations if c["num_instances"] == 1)
    multi = sum(1 for c in conversations if c["num_instances"] > 1)
    print(f"Saved {len(conversations)} conversations ({single} single, {multi} multi-instance) to {output_file}")


def main():
    process_split("validation")
    process_split("train")


if __name__ == "__main__":
    main()
