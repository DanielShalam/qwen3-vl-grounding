import json
import argparse
from pathlib import Path

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
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

def evaluate(predictions_file, threshold=0.5):
    """Evaluate predictions using IoU metric"""
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    ious = []
    correct = 0
    
    for pred in predictions:
        gt_box = pred["ground_truth_box"]
        pred_box = pred["predicted_box"]
        
        iou = calculate_iou(gt_box, pred_box)
        ious.append(iou)
        
        if iou >= threshold:
            correct += 1
    
    accuracy = correct / len(predictions) if predictions else 0
    mean_iou = sum(ious) / len(ious) if ious else 0
    
    print(f"Total samples: {len(predictions)}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Accuracy (IoU >= {threshold}): {accuracy:.4f} ({correct}/{len(predictions)})")
    
    # Save metrics
    metrics = {
        "total_samples": len(predictions),
        "mean_iou": mean_iou,
        "accuracy": accuracy,
        "threshold": threshold,
        "correct": correct
    }
    
    output_file = Path(predictions_file).parent / "metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to predictions JSON")
    parser.add_argument("--threshold", type=float, default=0.5, help="IoU threshold")
    args = parser.parse_args()
    
    evaluate(args.results, args.threshold)
