import json
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

def format_bbox_to_qwen(bbox, img_width, img_height):
    """Convert LVIS bbox [x, y, width, height] to Qwen format <box>(x1,y1),(x2,y2)</box>"""
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # Normalize to 0-1000 range (Qwen's coordinate system)
    x1_norm = int((x1 / img_width) * 1000)
    y1_norm = int((y1 / img_height) * 1000)
    x2_norm = int((x2 / img_width) * 1000)
    y2_norm = int((y2 / img_height) * 1000)
    
    return f"<box>({x1_norm},{y1_norm}),({x2_norm},{y2_norm})</box>"

def create_conversation(image_id, category, bbox_str, image_path):
    """Format into Qwen conversation structure"""
    return {
        "id": image_id,
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<img>{image_path}</img>\nLocate the {category} in this image."
            },
            {
                "from": "gpt",
                "value": f"The {category} is located at {bbox_str}"
            }
        ]
    }

def main():
    print("Loading LVIS dataset...")
    dataset = load_dataset("winvoker/lvis", split="validation")
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    conversations = []
    
    print("Processing annotations...")
    for item in tqdm(dataset):
        img_id = item["image_id"]
        img_width = item["width"]
        img_height = item["height"]
        
        for ann in item["annotations"]:
            bbox = ann["bbox"]
            category = ann["category"]
            
            bbox_str = format_bbox_to_qwen(bbox, img_width, img_height)
            conv = create_conversation(img_id, category, bbox_str, item["coco_url"])
            conversations.append(conv)
    
    # Save formatted dataset
    output_file = output_dir / "lvis_conversations.json"
    with open(output_file, "w") as f:
        json.dump(conversations, f, indent=2)
    
    print(f"Saved {len(conversations)} conversations to {output_file}")

if __name__ == "__main__":
    main()
