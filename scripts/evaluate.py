import json
import argparse
from pathlib import Path
from collections import defaultdict


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
    """Evaluate predictions using best-match IoU.

    Groups predictions by (image_id, predicted_box) and matches each prediction
    against all GT boxes for that image_id, taking the best IoU.
    This handles cases where multiple GT annotations exist for the same object category.
    """
    with open(predictions_file) as f:
        predictions = json.load(f)

    # Group GT boxes by image_id
    gt_by_image = defaultdict(list)
    for pred in predictions:
        gt_by_image[pred["id"]].append(pred["ground_truth_box"])

    # Deduplicate: one evaluation per unique (image_id, predicted_box)
    seen = set()
    ious = []
    correct = 0
    failed_parse = 0
    evaluated = 0

    for pred in predictions:
        pred_box = pred["predicted_box"]
        gt_box = pred["ground_truth_box"]

        if gt_box is None or pred_box is None:
            failed_parse += 1
            continue

        # Best-match: compare prediction against ALL GT boxes for this image
        all_gt = gt_by_image[pred["id"]]
        best_iou = max(calculate_iou(gt, pred_box) for gt in all_gt if gt is not None)

        # Deduplicate identical predictions for same image
        key = (pred["id"], tuple(pred_box))
        if key in seen:
            continue
        seen.add(key)

        ious.append(best_iou)
        evaluated += 1
        if best_iou >= threshold:
            correct += 1

    accuracy = correct / evaluated if evaluated else 0
    mean_iou = sum(ious) / len(ious) if ious else 0

    print(f"Total predictions: {len(predictions)}")
    print(f"Evaluated (deduplicated): {evaluated}")
    print(f"Failed to parse: {failed_parse}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Accuracy (IoU >= {threshold}): {accuracy:.4f} ({correct}/{evaluated})")

    metrics = {
        "total_predictions": len(predictions),
        "evaluated": evaluated,
        "failed_parse": failed_parse,
        "mean_iou": mean_iou,
        "accuracy": accuracy,
        "threshold": threshold,
        "correct": correct,
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
