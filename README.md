# Qwen3-VL Vision Grounding with LoRA Fine-tuning

Fine-tuning Qwen3-VL-8B-Instruct for vision grounding on the LVIS dataset using LoRA.

## Project Structure
```
├── scripts/
│   ├── prepare_lvis.py       # Download & format LVIS into Qwen conversation JSON
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
- **LVIS conversations**: `/efs/user_folders/dnshalam/datasets/lvis/{lvis_train.json,lvis_validation.json}`
- **HF cache**: `/efs/user_folders/dnshalam/datasets/.hf_cache/`

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
- Downloads LVIS dataset from HuggingFace (`winvoker/lvis`)
- Formats bounding boxes to Qwen format: `<box>(x1,y1),(x2,y2)</box>`
- Maps images to local COCO paths on EFS
- Outputs conversation JSONs per split

### 2. Zero-Shot Baseline
```bash
python scripts/run_inference.py --mode baseline
python scripts/evaluate.py --results results/baseline/predictions.json
```
Use `--limit N` to evaluate on a subset.

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

## Metrics
- Primary: IoU >= 0.5 accuracy
- Secondary: Mean IoU, precision/recall at various thresholds
