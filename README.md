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
| Qwen3-VL-8B-Instruct | LoRA 3ep (100K samples) | 1,000 | 0.6076 | 65.6% | 0.0% |
| Qwen3-VL-8B-Instruct | LoRA ~2.7ep (500K samples) | 1,000 | 0.6060 | 64.7% | 0.0% |

### Multi-Instance Evaluation (Hungarian Matching)

| Setup | Images | GT Boxes | Pred Boxes | Mean IoU | Recall@0.5 | Precision@0.5 | F1@0.5 |
|-------|--------|----------|------------|----------|------------|---------------|--------|
| Zero-shot (grouped, single-class) | 1,000 | 3,040 | 1,716 | 0.358 | 40.4% | 71.5% | 51.6% |
| Zero-shot (multiclass, per-image) | 500 | 5,154 | 3,330 | 0.365 | 41.1% | 63.5% | 49.9% |

## Metrics
- Primary: IoU >= 0.5 accuracy
- Secondary: Mean IoU, precision/recall at various thresholds
