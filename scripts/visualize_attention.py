"""
Visualize attention maps during bounding box coordinate generation.

Two visualization modes:
  1. --mode layers: All layers (head-averaged) for each coordinate - shows attention evolution through depth
  2. --mode heads:  All heads of a specific layer for each coordinate - shows head specialization

For each coordinate (x1, y1, x2, y2), shows:
  - Attention breakdown (image vs text vs generated tokens)
  - Spatial attention heatmap over the image
"""

import json
import re
import torch
import yaml
import argparse
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
        attn_implementation="eager",
    )
    model.eval()
    return model, processor


def get_image_token_mask(input_ids, processor):
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return (input_ids[0] == image_token_id).cpu().numpy()


def generate_with_attention(model, inputs, max_new_tokens=128):
    """Generate tokens one by one, capturing ALL layer/head attention at each step."""
    input_ids = inputs["input_ids"].clone()
    attention_records = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in inputs.items() if k != "input_ids"},
                input_ids=input_ids,
                output_attentions=True,
            )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Store per-layer, per-head attention FROM last token TO all others
        # outputs.attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
        per_layer = []
        for layer_attn in outputs.attentions:
            # (num_heads, seq_len) - attention from last token to all positions
            per_layer.append(layer_attn[0, :, -1, :].cpu().float().numpy())

        attention_records.append({
            "token_id": next_token.item(),
            "per_layer": per_layer,  # list[num_layers] of (num_heads, seq_len)
        })

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == model.config.eos_token_id:
            break

    return input_ids, attention_records


def find_coordinate_steps(attention_records, processor):
    token_ids = [r["token_id"] for r in attention_records]
    tokens = [processor.tokenizer.decode(tid) for tid in token_ids]
    full_text = "".join(tokens)

    match = re.search(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', full_text)
    if not match:
        print(f"No bbox found in: {full_text[:200]}")
        return None

    coord_names = ["x1", "y1", "x2", "y2"]
    coord_steps = {}
    char_pos = 0
    for step_idx, token_str in enumerate(tokens):
        token_start = char_pos
        token_end = char_pos + len(token_str)
        for i, name in enumerate(coord_names):
            if name not in coord_steps and token_start <= match.start(i + 1) < token_end:
                coord_steps[name] = step_idx
        char_pos = token_end

    return coord_steps


def compute_attention_breakdown(attn_vector, image_mask, num_input_tokens):
    total = attn_vector.sum()
    if total == 0:
        return {"image": 0, "text_prompt": 0, "generated": 0}
    image_attn = attn_vector[:num_input_tokens][image_mask[:num_input_tokens]].sum() / total
    text_attn = attn_vector[:num_input_tokens][~image_mask[:num_input_tokens]].sum() / total
    generated_attn = attn_vector[num_input_tokens:].sum() / total
    return {"image": float(image_attn), "text_prompt": float(text_attn), "generated": float(generated_attn)}


def create_spatial_heatmap(attn_vector, image_mask, img_w, img_h):
    image_attn = attn_vector[:len(image_mask)][image_mask[:len(attn_vector)]]
    n = image_attn.shape[0]
    if n == 0:
        return np.zeros((img_h, img_w))
    side = int(np.sqrt(n))
    if side * side != n:
        side = int(np.ceil(np.sqrt(n)))
        image_attn = np.pad(image_attn, (0, side * side - n))
    heatmap = image_attn.reshape(side, side)
    return np.array(Image.fromarray(heatmap).resize((img_w, img_h), Image.BILINEAR))


def draw_gt_box(ax, gt_coords, img_w, img_h):
    if gt_coords:
        gx1, gy1, gx2, gy2 = [c / 1000 for c in gt_coords]
        rect = patches.Rectangle(
            (gx1 * img_w, gy1 * img_h), (gx2 - gx1) * img_w, (gy2 - gy1) * img_h,
            linewidth=2, edgecolor="lime", facecolor="none", linestyle="--")
        ax.add_patch(rect)


def parse_gt_coords(item):
    match = re.search(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', item["conversations"][1]["value"])
    return [int(match.group(i)) for i in range(1, 5)] if match else None


def prepare_inference(model, processor, item):
    image_path = item["image"]
    prompt_text = re.sub(r'<img>.*?</img>\n', '', item["conversations"][0]["value"])
    image = Image.open(image_path).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)

    return image, prompt_text, inputs


# ── Mode 1: All layers, head-averaged ──

def visualize_layers(model, processor, item, output_path):
    """Plot head-averaged attention for every layer at each coordinate step."""
    image, prompt_text, inputs = prepare_inference(model, processor, item)
    num_input = inputs["input_ids"].shape[1]
    image_mask = get_image_token_mask(inputs["input_ids"], processor)
    gt_coords = parse_gt_coords(item)

    output_ids, attn_records = generate_with_attention(model, inputs)
    generated_text = processor.decode(output_ids[0][num_input:], skip_special_tokens=False)
    coord_steps = find_coordinate_steps(attn_records, processor)
    if not coord_steps:
        return

    num_layers = len(attn_records[0]["per_layer"])
    coord_names = ["x1", "y1", "x2", "y2"]

    # ── Breakdown plot: layers x coords ──
    fig, axes = plt.subplots(num_layers, 4, figsize=(16, 2.5 * num_layers))
    if num_layers == 1:
        axes = axes[np.newaxis, :]

    for col, name in enumerate(coord_names):
        if name not in coord_steps:
            continue
        step = coord_steps[name]
        for layer_idx in range(num_layers):
            attn = attn_records[step]["per_layer"][layer_idx].mean(axis=0)  # avg heads
            bd = compute_attention_breakdown(attn, image_mask, num_input)
            ax = axes[layer_idx, col]
            bars = ax.bar(bd.keys(), bd.values(), color=["#2196F3", "#FF9800", "#4CAF50"])
            ax.set_ylim(0, 1)
            if layer_idx == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=9)
            ax.tick_params(labelsize=7)
            for bar, val in zip(bars, bd.values()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.0%}", ha="center", fontsize=7)

    fig.suptitle(f"Attention Breakdown per Layer\n{prompt_text}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path.with_name(output_path.stem + "_layers_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Spatial heatmap: sample of layers ──
    sample_layers = np.linspace(0, num_layers - 1, min(6, num_layers), dtype=int)
    fig, axes = plt.subplots(len(sample_layers), 4, figsize=(16, 3 * len(sample_layers)))
    if len(sample_layers) == 1:
        axes = axes[np.newaxis, :]

    for col, name in enumerate(coord_names):
        if name not in coord_steps:
            continue
        step = coord_steps[name]
        for row, layer_idx in enumerate(sample_layers):
            attn = attn_records[step]["per_layer"][layer_idx].mean(axis=0)
            heatmap = create_spatial_heatmap(attn, image_mask, image.width, image.height)
            ax = axes[row, col]
            ax.imshow(image)
            ax.imshow(heatmap, alpha=0.5, cmap="hot")
            draw_gt_box(ax, gt_coords, image.width, image.height)
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=10)
                ax.yaxis.set_visible(True)

    fig.suptitle(f"Spatial Attention per Layer\n{prompt_text}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path.with_name(output_path.stem + "_layers_spatial.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved layer visualizations to {output_path.parent}")


# ── Mode 2: All heads of one layer ──

def visualize_heads(model, processor, item, output_path, layer_idx=-1):
    """Plot per-head attention for a single layer at each coordinate step."""
    image, prompt_text, inputs = prepare_inference(model, processor, item)
    num_input = inputs["input_ids"].shape[1]
    image_mask = get_image_token_mask(inputs["input_ids"], processor)
    gt_coords = parse_gt_coords(item)

    output_ids, attn_records = generate_with_attention(model, inputs)
    generated_text = processor.decode(output_ids[0][num_input:], skip_special_tokens=False)
    coord_steps = find_coordinate_steps(attn_records, processor)
    if not coord_steps:
        return

    num_layers = len(attn_records[0]["per_layer"])
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    num_heads = attn_records[0]["per_layer"][layer_idx].shape[0]
    coord_names = ["x1", "y1", "x2", "y2"]

    # ── Breakdown: heads x coords ──
    fig, axes = plt.subplots(num_heads, 4, figsize=(16, 1.8 * num_heads))
    if num_heads == 1:
        axes = axes[np.newaxis, :]

    for col, name in enumerate(coord_names):
        if name not in coord_steps:
            continue
        step = coord_steps[name]
        layer_attn = attn_records[step]["per_layer"][layer_idx]  # (num_heads, seq_len)
        for head_idx in range(num_heads):
            attn = layer_attn[head_idx]
            bd = compute_attention_breakdown(attn, image_mask, num_input)
            ax = axes[head_idx, col]
            bars = ax.bar(bd.keys(), bd.values(), color=["#2196F3", "#FF9800", "#4CAF50"])
            ax.set_ylim(0, 1)
            if head_idx == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"H{head_idx}", fontsize=9)
            ax.tick_params(labelsize=6)
            for bar, val in zip(bars, bd.values()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.0%}", ha="center", fontsize=6)

    fig.suptitle(f"Attention Breakdown per Head (Layer {layer_idx})\n{prompt_text}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path.with_name(output_path.stem + f"_heads_L{layer_idx}_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Spatial heatmap: sample of heads ──
    sample_heads = np.linspace(0, num_heads - 1, min(8, num_heads), dtype=int)
    fig, axes = plt.subplots(len(sample_heads), 4, figsize=(16, 3 * len(sample_heads)))
    if len(sample_heads) == 1:
        axes = axes[np.newaxis, :]

    for col, name in enumerate(coord_names):
        if name not in coord_steps:
            continue
        step = coord_steps[name]
        layer_attn = attn_records[step]["per_layer"][layer_idx]
        for row, head_idx in enumerate(sample_heads):
            attn = layer_attn[head_idx]
            heatmap = create_spatial_heatmap(attn, image_mask, image.width, image.height)
            ax = axes[row, col]
            ax.imshow(image)
            ax.imshow(heatmap, alpha=0.5, cmap="hot")
            draw_gt_box(ax, gt_coords, image.width, image.height)
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Head {head_idx}", fontsize=10)
                ax.yaxis.set_visible(True)

    fig.suptitle(f"Spatial Attention per Head (Layer {layer_idx})\n{prompt_text}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path.with_name(output_path.stem + f"_heads_L{layer_idx}_spatial.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved head visualizations for layer {layer_idx} to {output_path.parent}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--mode", choices=["layers", "heads", "both"], default="both")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index for heads mode (-1 = last)")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    model, processor = load_model(args.config)

    with open(DATA_DIR / "lvis_validation.json") as f:
        data = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num_samples, len(data))):
        print(f"\nSample {i + 1}/{args.num_samples}")
        out = OUTPUT_DIR / f"sample_{i}"
        if args.mode in ("layers", "both"):
            visualize_layers(model, processor, data[i], out)
        if args.mode in ("heads", "both"):
            visualize_heads(model, processor, data[i], out, layer_idx=args.layer)


if __name__ == "__main__":
    main()
