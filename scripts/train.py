import json
import yaml
from pathlib import Path
from PIL import Image
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

DATA_DIR = Path("/efs/user_folders/dnshalam/datasets/lvis")


def load_config(config_path="configs/lora_config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(data_file):
    """Load LVIS conversation JSON and convert to Unsloth vision format."""
    with open(data_file) as f:
        data = json.load(f)

    dataset = []
    for item in data:
        image_path = item["image"]
        human_msg = item["conversations"][0]["value"]
        # Extract prompt text without <img> tags
        import re
        prompt_text = re.sub(r'<img>.*?</img>\n', '', human_msg)
        answer = item["conversations"][1]["value"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            continue

        dataset.append({"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer},
            ]},
        ]})

    print(f"Loaded {len(dataset)} samples")
    return dataset


def main():
    config = load_config()
    model_name = config["model_name"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    print(f"Loading model: {model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        max_seq_length=config["data"]["max_length"],
        load_in_4bit=True,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        random_state=3407,
        use_rslora=False,
        target_modules="all-linear",
    )

    print("Loading training data...")
    train_dataset = load_dataset(DATA_DIR / "lvis_train.json")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            warmup_steps=train_cfg["warmup_steps"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            fp16=not train_cfg.get("bf16", False),
            bf16=train_cfg.get("bf16", False),
            optim=train_cfg["optim"],
            dataloader_num_workers=train_cfg["dataloader_num_workers"],
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(f"{train_cfg['output_dir']}/final")
    tokenizer.save_pretrained(f"{train_cfg['output_dir']}/final")
    print("Training complete!")


if __name__ == "__main__":
    main()
