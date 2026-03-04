# Qwen3-VL Vision Grounding with LoRA Fine-tuning

Fine-tuning Qwen3-VL for vision grounding on the LVIS dataset using LoRA.

## Project Structure
```
├── data/                  # LVIS dataset and formatted conversations
├── scripts/               # Executable scripts for each pipeline stage
├── src/                   # Core modules
├── configs/               # LoRA and training configurations
├── results/
│   ├── baseline/         # Zero-shot inference results
│   ├── finetuned/        # Post-training results
│   └── analysis/         # Qualitative failure analysis
└── checkpoints/          # LoRA adapters and merged weights
```

## Pipeline

### 1. Data Preparation
```bash
python scripts/prepare_lvis.py
```
- Downloads LVIS dataset from HuggingFace
- Formats bounding boxes to Qwen format: `<box>(x1,y1),(x2,y2)</box>`
- Converts to conversation JSON format

### 2. Zero-Shot Baseline
```bash
python scripts/run_inference.py --mode baseline
python scripts/evaluate.py --results results/baseline/predictions.json
```
- Runs zero-shot inference on validation set
- Computes IoU-based accuracy (threshold: IoU >= 0.5)

### 3. LoRA Fine-tuning
```bash
python scripts/train.py --config configs/lora_config.yaml
```
- LoRA rank: r=16 or r=32
- Targets: ViT linear layers + LLM QKV projections
- Training: 2-3 epochs

### 4. Post-Training Evaluation
```bash
python scripts/merge_lora.py
python scripts/run_inference.py --mode finetuned
python scripts/evaluate.py --results results/finetuned/predictions.json
```

### 5. Failure Analysis
```bash
python scripts/analyze_failures.py
```
- Boundary regression errors vs. hallucinations
- Visualizations and error categorization

## Setup
```bash
conda activate /efs/user_folders/dnshalam/envs/<your-env-name>
pip install -r requirements.txt
```

## Metrics
- Primary: IoU >= 0.5 accuracy
- Secondary: Mean IoU, precision/recall at various thresholds
