"""
Prepare LVIS dataset with multi-class grouping per image.
Groups all annotations by image, lists target categories in the prompt,
and outputs all bboxes with labels in one response.
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
    output_file = OUTPUT_DIR / f"lvis_{split_name}_multiclass.json"
    config = SPLIT_CONFIG[split_name]

    print(f"\nLoading LVIS {split_name} annotations...")
    with open(config["annotations"]) as f:
        lvis_data = json.load(f)

    images = {img["id"]: img for img in lvis_data["images"]}
    categories = {cat["id"]: cat["name"] for cat in lvis_data["categories"]}

    # Group annotations by image_id
    by_image = defaultdict(list)
    for ann in lvis_data["annotations"]:
        by_image[ann["image_id"]].append(ann)

    conversations = []
    print(f"Processing {len(by_image)} images...")
    for img_id, anns in tqdm(by_image.items()):
        img = images[img_id]
        img_w, img_h = img["width"], img["height"]
        filename = img["coco_url"].split("/")[-1]
        image_path = str(config["images_dir"] / filename)

        cat_names = sorted(set(categories[a["category_id"]] for a in anns))
        bboxes = [
            {"bbox_2d": format_bbox(ann["bbox"], img_w, img_h), "label": categories[ann["category_id"]]}
            for ann in anns
        ]

        cat_list = ", ".join(cat_names)
        prompt = f"<image>\nLocate all instances of {cat_list} in this image and output the bbox coordinates in JSON format."
        answer = json.dumps(bboxes)

        conversations.append({
            "id": img_id,
            "image": image_path,
            "categories": cat_names,
            "num_objects": len(bboxes),
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(conversations, f, indent=2)

    avg_cats = sum(len(c["categories"]) for c in conversations) / len(conversations)
    avg_objs = sum(c["num_objects"] for c in conversations) / len(conversations)
    print(f"Saved {len(conversations)} images (avg {avg_cats:.1f} categories, {avg_objs:.1f} objects per image) to {output_file}")


def main():
    process_split("validation")
    process_split("train")


if __name__ == "__main__":
    main()
