import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen3ForConditionalGeneration, AutoProcessor
from peft import PeftModel

def parse_bbox_from_response(response):
    """Extract bbox coordinates from model response"""
    import re
    pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    match = re.search(pattern, response)
    if match:
        return [int(match.group(i)) for i in range(1, 5)]
    return None

def run_inference(mode="baseline", config_path="configs/lora_config.yaml"):
    """Run inference on validation set"""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_name = config["model_name"]
    print(f"Loading model: {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if mode == "finetuned":
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, "checkpoints/final")
    
    # Load validation data
    with open("data/lvis_conversations.json") as f:
        data = json.load(f)
    
    predictions = []
    
    print(f"Running {mode} inference...")
    for item in tqdm(data[:100]):  # Limit for validation
        messages = item["conversations"]
        
        inputs = processor(
            text=messages[0]["value"],
            images=item["image"],
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        
        response = processor.decode(outputs[0], skip_special_tokens=False)
        pred_box = parse_bbox_from_response(response)
        gt_box = parse_bbox_from_response(messages[1]["value"])
        
        predictions.append({
            "id": item["id"],
            "ground_truth_box": gt_box,
            "predicted_box": pred_box,
            "response": response
        })
    
    # Save predictions
    output_dir = Path(f"results/{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "predictions.json"
    
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetuned"], default="baseline")
    parser.add_argument("--config", default="configs/lora_config.yaml")
    args = parser.parse_args()
    
    run_inference(args.mode, args.config)
