import torch
import yaml
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

def merge_lora_weights():
    """Merge LoRA adapter weights into base model"""
    with open("configs/lora_config.yaml") as f:
        config = yaml.safe_load(f)
    
    model_name = config["model_name"]
    print(f"Loading base model: {model_name}")
    
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "checkpoints/final")
    
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    print("Saving merged model...")
    merged_model.save_pretrained("checkpoints/merged")
    
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained("checkpoints/merged")
    
    print("Merge complete! Saved to checkpoints/merged")

if __name__ == "__main__":
    merge_lora_weights()
