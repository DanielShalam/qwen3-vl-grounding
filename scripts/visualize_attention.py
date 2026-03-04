"""
Visualize attention maps during bounding box coordinate generation.
For each coordinate (x1, y1, x2, y2), shows:
  - How much attention goes to image tokens vs text tokens
  - Spatial attention heatmap over the image
"""

import json
import re
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

DATA_DIR = Path("/efs/user_folders/dnshalam/datasets/lvis")
OUTPUT_DIR = Path("results/analysis/attention_maps")


def load_model(config_path="configs/lora_config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_name = config["model_name"]
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager",  # Need eager attention for weights
    )
    model.eval()
    return model, processor


def get_image_token_mask(input_ids, processor):
    """Identify which token positions correspond to image tokens."""
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return (input_ids[0] == image_token_id).cpu().numpy()


def generate_with_attention(model, inputs, max_new_tokens=128):
    """Generate tokens one by one, capturing attention at each step."""
    input_ids = inputs["input_ids"].clone()
    attention_records = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in inputs.items() if k != "input_ids"},
                input_ids=input_ids,
                output_attentions=True,
            )

        # Get next token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded = model.config._name_or_path  # placeholder
        token_str = inputs.get("_processor", None)

        # Store attention from last layer for the last generated token
        # Shape: (batch, num_heads, seq_len, seq_len)
        last_layer_attn = outputs.attentions[-1]
        # Average over heads, get attention FROM the last token TO all others
        attn_from_last = last_layer_attn[0, :, -1, :].mean(dim=0).cpu().float().numpy()

        attention_records.append({
            "token_id": next_token.item(),
            "attention": attn_from_last,
        })

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Stop on EOS
        if next_token.item() == model.config.eos_token_id:
            break

    return input_ids, attention_records


def find_coordinate_steps(attention_records, processor):
    """Find which generation steps correspond to bbox coordinate digits."""
    token_ids = [r["token_id"] for r in attention_records]
    tokens = [processor.tokenizer.decode(tid) for tid in token_ids]
    full_text = "".join(tokens)

    # Find bbox_2d pattern in generated text
    match = re.search(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', full_text)
    if not match:
        print(f"No bbox found in: {full_text[:200]}")
        return None

    # Map character positions back to token steps
    coord_names = ["x1", "y1", "x2", "y2"]
    coord_steps = {}
    char_pos = 0

    for step_idx, token_str in enumerate(tokens):
        token_start = char_pos
        token_end = char_pos + len(token_str)

        for i, name in enumerate(coord_names):
            coord_start = match.start(i + 1)
            coord_end = match.end(i + 1)
            # First token of each coordinate
            if token_start <= coord_start < token_end and name not in coord_steps:
                coord_steps[name] = step_idx

        char_pos = token_end

    return coord_steps


def compute_attention_breakdown(attn_vector, image_mask, num_input_tokens):
    """Split attention into image vs text vs generated portions."""
    total = attn_vector.sum()
    image_attn = attn_vector[:num_input_tokens][image_mask[:num_input_tokens]].sum() / total
    text_attn = attn_vector[:num_input_tokens][~image_mask[:num_input_tokens]].sum() / total
    generated_attn = attn_vector[num_input_tokens:].sum() / total
    return {
        "image": float(image_attn),
        "text_prompt": float(text_attn),
        "generated": float(generated_attn),
    }


def create_spatial_heatmap(attn_vector, image_mask, grid_h, grid_w):
    """Reshape image token attention into spatial heatmap."""
    image_attn = attn_vector[:len(image_mask)][image_mask[:len(attn_vector)]]
    # Qwen3-VL image tokens form a grid
    n_image_tokens = image_attn.shape[0]
    if n_image_tokens == 0:
        return np.zeros((grid_h, grid_w))

    # Try to reshape to grid
    if n_image_tokens == grid_h * grid_w:
        return image_attn.reshape(grid_h, grid_w)

    # Fallback: interpolate
    side = int(np.sqrt(n_image_tokens))
    if side * side == n_image_tokens:
        return image_attn.reshape(side, side)

    return image_attn[:grid_h * grid_w].reshape(grid_h, grid_w) if n_image_tokens >= grid_h * grid_w else np.zeros((grid_h, grid_w))


def visualize_sample(model, processor, item, output_path):
    """Run inference on one sample and visualize attention at coordinate steps."""
    image_path = item["image"]
    prompt_text = re.sub(r'<img>.*?</img>\n', '', item["conversations"][0]["value"])
    gt_box = re.search(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', item["conversations"][1]["value"])
    gt_coords = [int(gt_box.group(i)) for i in range(1, 5)] if gt_box else None

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)

    num_input_tokens = inputs["input_ids"].shape[1]
    image_mask = get_image_token_mask(inputs["input_ids"], processor)

    print(f"  Input tokens: {num_input_tokens}, Image tokens: {image_mask.sum()}")

    # Generate with attention
    output_ids, attention_records = generate_with_attention(model, inputs)
    generated_text = processor.decode(output_ids[0][num_input_tokens:], skip_special_tokens=False)
    print(f"  Generated: {generated_text[:150]}")

    # Find coordinate generation steps
    coord_steps = find_coordinate_steps(attention_records, processor)
    if coord_steps is None:
        print("  Could not find bbox coordinates in output, skipping.")
        return

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    coord_names = ["x1", "y1", "x2", "y2"]

    for col, name in enumerate(coord_names):
        if name not in coord_steps:
            continue

        step = coord_steps[name]
        attn = attention_records[step]["attention"]

        # Top row: attention breakdown bar chart
        breakdown = compute_attention_breakdown(attn, image_mask, num_input_tokens)
        ax = axes[0, col]
        bars = ax.bar(breakdown.keys(), breakdown.values(), color=["#2196F3", "#FF9800", "#4CAF50"])
        ax.set_title(f"{name} attention breakdown", fontsize=12)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, breakdown.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.1%}", ha="center", fontsize=10)

        # Bottom row: spatial heatmap over image
        ax = axes[1, col]
        ax.imshow(image)
        n_img = int(image_mask.sum())
        side = int(np.sqrt(n_img)) if n_img > 0 else 1
        heatmap = create_spatial_heatmap(attn, image_mask, side, side)
        heatmap_resized = np.array(Image.fromarray(heatmap).resize(
            (image.width, image.height), Image.BILINEAR))
        ax.imshow(heatmap_resized, alpha=0.5, cmap="hot")
        ax.set_title(f"{name} spatial attention", fontsize=12)
        ax.axis("off")

        # Draw GT box if available
        if gt_coords:
            gx1, gy1, gx2, gy2 = [c / 1000 for c in gt_coords]
            rect = patches.Rectangle(
                (gx1 * image.width, gy1 * image.height),
                (gx2 - gx1) * image.width, (gy2 - gy1) * image.height,
                linewidth=2, edgecolor="lime", facecolor="none", linestyle="--")
            ax.add_patch(rect)

    fig.suptitle(f"Attention Analysis: {prompt_text}\nGenerated: {generated_text[:100]}", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    model, processor = load_model(args.config)

    with open(DATA_DIR / "lvis_validation.json") as f:
        data = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num_samples, len(data))):
        print(f"\nSample {i + 1}/{args.num_samples}")
        visualize_sample(model, processor, data[i], OUTPUT_DIR / f"attention_sample_{i}.png")


if __name__ == "__main__":
    main()
