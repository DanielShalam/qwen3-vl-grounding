import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def categorize_failure(pred, gt_box, pred_box, iou):
    """Categorize failure type"""
    if pred_box is None:
        return "hallucination_no_box"
    
    if iou < 0.3:
        return "severe_localization_error"
    elif iou < 0.5:
        return "boundary_regression_error"
    else:
        return "success"

def analyze_failures(predictions_file):
    """Analyze failure modes"""
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    failure_categories = {
        "hallucination_no_box": [],
        "severe_localization_error": [],
        "boundary_regression_error": [],
        "success": []
    }
    
    for pred in predictions:
        gt_box = pred["ground_truth_box"]
        pred_box = pred["predicted_box"]
        
        if pred_box is None:
            iou = 0.0
        else:
            iou = calculate_iou(gt_box, pred_box)
        
        category = categorize_failure(pred, gt_box, pred_box, iou)
        failure_categories[category].append({
            "id": pred["id"],
            "iou": iou,
            "response": pred["response"]
        })
    
    # Print summary
    print("\n=== Failure Analysis ===")
    for category, items in failure_categories.items():
        print(f"{category}: {len(items)} ({len(items)/len(predictions)*100:.1f}%)")
    
    # Save detailed analysis
    output_dir = Path(predictions_file).parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "failure_analysis.json", "w") as f:
        json.dump(failure_categories, f, indent=2)
    
    print(f"\nDetailed analysis saved to {output_dir / 'failure_analysis.json'}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    categories = list(failure_categories.keys())
    counts = [len(failure_categories[cat]) for cat in categories]
    
    sns.barplot(x=categories, y=counts)
    plt.title("Failure Mode Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "failure_distribution.png")
    print(f"Plot saved to {output_dir / 'failure_distribution.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to predictions JSON")
    args = parser.parse_args()
    
    analyze_failures(args.results)
