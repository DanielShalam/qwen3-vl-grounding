import yaml
import torch
from pathlib import Path
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def prepare_dataset(data_file, processor):
    import json
    with open(data_file) as f:
        data = json.load(f)
    
    def preprocess(examples):
        texts = [ex["conversations"][0]["value"] for ex in examples]
        images = [ex["image"] for ex in examples]
        labels = [ex["conversations"][1]["value"] for ex in examples]
        
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs["labels"] = processor.tokenizer(labels, return_tensors="pt", padding=True)["input_ids"]
        return inputs
    
    return Dataset.from_list(data)

def main():
    config = load_config("configs/lora_config.yaml")
    
    print(f"Loading model: {config['model_name']}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(config["model_name"])
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    train_dataset = prepare_dataset(config["data"]["train_file"], processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        fp16=config["training"]["fp16"],
        optim=config["training"]["optim"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save final model
    model.save_pretrained(f"{config['training']['output_dir']}/final")
    print("Training complete!")

if __name__ == "__main__":
    main()
