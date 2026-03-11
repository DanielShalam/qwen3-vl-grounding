# Qwen3-VL Vision Grounding with LoRA Fine-tuning

Fine-tuning Qwen3-VL-8B-Instruct for vision grounding on the LVIS dataset using LoRA.

## Project Structure
```
├── scripts/
│   ├── prepare_lvis.py       # Format LVIS annotations into conversation JSON
│   ├── run_inference.py       # Zero-shot / fine-tuned inference
│   ├── evaluate.py            # IoU-based evaluation
│   ├── train.py               # LoRA fine-tuning
│   ├── merge_lora.py          # Merge LoRA weights into base model
│   └── analyze_failures.py    # Qualitative failure analysis
├── configs/
│   └── lora_config.yaml       # LoRA and training configuration
├── results/
│   ├── baseline/              # Zero-shot inference results
│   ├── finetuned/             # Post-training results
│   └── analysis/              # Failure analysis outputs
└── checkpoints/               # LoRA adapters and merged weights
```

## Data (EFS)
Data is stored on shared EFS, not in the repo:
- **COCO images**: `/efs/user_folders/dnshalam/datasets/coco2017/{train2017,val2017}/`
- **LVIS raw annotations**: `/efs/user_folders/dnshalam/datasets/lvis/{lvis_v1_train.json,lvis_v1_val.json}`
- **Formatted conversations**: `/efs/user_folders/dnshalam/datasets/lvis/{lvis_train.json,lvis_validation.json}`

## Setup
```bash
conda activate /efs/user_folders/dnshalam/envs/qwen3-vl
pip install -r requirements.txt
```

## Pipeline

### 1. Data Preparation
```bash
python scripts/prepare_lvis.py
```
- Reads LVIS v1 annotation JSONs (downloaded from Facebook's LVIS release)
- Normalizes bounding boxes to Qwen's 0-1000 coordinate system
- Maps images to local COCO paths on EFS
- Outputs conversation JSONs per split (train: 1.27M, val: 244K annotations)

### 2. Zero-Shot Baseline
```bash
python scripts/run_inference.py --mode baseline
python scripts/evaluate.py --results results/baseline/predictions.json
```
Use `--limit N` to evaluate on a subset. Model outputs bboxes in `{"bbox_2d": [x1, y1, x2, y2]}` format.

### 3. LoRA Fine-tuning
```bash
python scripts/train.py
```
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- LoRA targets: ViT linear layers + LLM QKV projections
- Config: `configs/lora_config.yaml`

### 4. Post-Training Evaluation
```bash
python scripts/merge_lora.py
python scripts/run_inference.py --mode finetuned
python scripts/evaluate.py --results results/finetuned/predictions.json
```

### 5. Failure Analysis
```bash
python scripts/analyze_failures.py --results results/finetuned/predictions.json
```

## Results

| Model | Mode | Samples | Mean IoU | Acc (IoU≥0.5) | Failed Parse |
|-------|------|---------|----------|---------------|--------------|
| Qwen3-VL-8B-Instruct | Zero-shot | 1,000 | 0.5973 | 63.3% | 0.5% |
| Qwen3-VL-8B-Instruct | LoRA 3ep, lr=2e-4 (100K) | 1,000 | 0.6076 | 65.6% | 0.0% |
| Qwen3-VL-8B-Instruct | LoRA ~2.7ep, lr=2e-4 (500K) | 1,000 | 0.6060 | 64.7% | 0.0% |
| Qwen3-VL-8B-Instruct | LoRA 3ep, lr=1e-6, Qwen-aligned (100K) | 1,000 | 0.6122 | 65.3% | 0.0% |

### Multi-Instance Evaluation (Hungarian Matching)

Evaluation uses per-category Hungarian matching to optimally assign predicted boxes to GT boxes.

**Impact of max generation tokens:** With many objects per image (~12.5 avg), the model's response
can be truncated if `max_new_tokens` is too low. This directly affects recall since truncated
responses miss objects at the end.

| Setup | max_tokens | Pred Boxes | Mean IoU | Img Acc@0.5 | Recall@0.5 | Precision@0.5 | F1@0.5 |
|-------|-----------|-----------|----------|-------------|------------|---------------|--------|
| Zero-shot multiclass (500 imgs) | 1024 | 3,330 | 36.5% | 65.1% | 41.1% | 63.5% | 49.9% |
| Zero-shot multiclass (500 imgs) | 4096 | 4,469 | 38.2% | 65.0% | 42.1% | 48.6% | 45.1% |
| Zero-shot grouped single-class (1K) | 1024 | 1,716 | 35.8% | - | 40.4% | 71.5% | 51.6% |

Key observations:
- Increasing tokens from 1024→4096 yields 34% more predictions but only marginal recall gain (+1%)
- The model naturally outputs ~30 boxes before stopping, even with higher token budget
- Single-class prompts achieve higher precision (71.5%) since the model focuses on one category
- Multiclass prompts find more objects overall but with lower precision due to cross-class confusion

## Metrics
- Primary: IoU >= 0.5 accuracy
- Secondary: Mean IoU, precision/recall at various thresholds
