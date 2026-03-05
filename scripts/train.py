import json
import yaml
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

DATA_DIR = Path("/efs/user_folders/dnshalam/datasets/lvis")


def load_config(config_path="configs/lora_config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


class LVISDataset(Dataset):
    """Lazy-loading dataset - only opens images when accessed."""

    def __init__(self, data_file, max_samples=None):
        with open(data_file) as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = item["conversations"][0]["value"]
        answer = item["conversations"][1]["value"]
        image = Image.open(item["image"]).convert("RGB")

        return {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer},
            ]},
        ]}


def main(config_path="configs/lora_config.yaml"):
    config = load_config(config_path)
    model_name = config["model_name"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    print(f"Loading model: {model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        max_seq_length=config["data"]["max_length"],
        load_in_4bit=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=0,
        bias=lora_cfg["bias"],
        random_state=3407,
        use_rslora=False,
        target_modules="all-linear",
    )

    print("Loading training data...")
    train_dataset = LVISDataset(DATA_DIR / "lvis_train.json", max_samples=config["data"].get("max_samples"))

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
            learning_rate=float(train_cfg["learning_rate"]),
            warmup_steps=train_cfg["warmup_steps"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            bf16=True,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    args = parser.parse_args()
    main(args.config)
