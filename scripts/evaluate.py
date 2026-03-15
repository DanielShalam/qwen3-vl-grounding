"""Evaluate grounding predictions.

Supports two modes:
1. Legacy (single-box): matches pred against all GT boxes for same image_id
2. Label-aware (grouped): matches pred against GT boxes of the asked category
   using lvis_validation_grouped.json as GT source
"""
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def evaluate_grouped(predictions_file, threshold=0.5):
    """Label-aware evaluation for grouped predictions."""
    with open(predictions_file) as f:
        predictions = json.load(f)

    ious, correct, total, failed = [], 0, 0, 0
    label_matches = 0

    for p in predictions:
        asked = p["asked_category"]
        gt_boxes = p["gt_boxes"]
        preds = p.get("pred_bboxes", [])

        if not preds:
            failed += 1
            continue

        for pb in preds:
            box = pb["box"]
            label = pb.get("label")
            if label:
                label_match = label.lower() == asked.lower()
            else:
                label_match = False
            if label_match:
                label_matches += 1

            # Best IoU against GT boxes of asked category
            best = max((calculate_iou(gb, box) for gb in gt_boxes), default=0.0)
            ious.append(best)
            total += 1
            if best >= threshold:
                correct += 1

    mean_iou = sum(ious) / len(ious) if ious else 0
    acc = correct / total if total else 0

    metrics = {
        "total": total, "failed": failed, "label_matches": label_matches,
        "mean_iou": mean_iou, "accuracy": acc, "correct": correct,
        "threshold": threshold,
    }

    print(f"Total predictions: {total}, No predictions: {failed}")
    print(f"Label match: {label_matches}/{total}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Accuracy@{threshold}: {acc:.4f} ({correct}/{total})")

    output_file = Path(predictions_file).parent / "metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {output_file}")


def evaluate_legacy(predictions_file, threshold=0.5):
    """Legacy single-box evaluation (backward compatible)."""
    with open(predictions_file) as f:
        predictions = json.load(f)

    gt_by_image = defaultdict(list)
    for pred in predictions:
        gt_by_image[pred["id"]].append(pred["ground_truth_box"])

    seen = set()
    ious, correct, failed, evaluated = [], 0, 0, 0

    for pred in predictions:
        pred_box, gt_box = pred["predicted_box"], pred["ground_truth_box"]
        if gt_box is None or pred_box is None:
            failed += 1
            continue
        all_gt = gt_by_image[pred["id"]]
        best_iou = max(calculate_iou(gt, pred_box) for gt in all_gt if gt is not None)
        key = (pred["id"], tuple(pred_box))
        if key in seen:
            continue
        seen.add(key)
        ious.append(best_iou)
        evaluated += 1
        if best_iou >= threshold:
            correct += 1

    acc = correct / evaluated if evaluated else 0
    mean_iou = sum(ious) / len(ious) if ious else 0

    metrics = {"total_predictions": len(predictions), "evaluated": evaluated,
               "failed_parse": failed, "mean_iou": mean_iou, "accuracy": acc,
               "threshold": threshold, "correct": correct}

    print(f"Total: {len(predictions)}, Evaluated: {evaluated}, Failed: {failed}")
    print(f"Mean IoU: {mean_iou:.4f}, Accuracy@{threshold}: {acc:.4f} ({correct}/{evaluated})")

    output_file = Path(predictions_file).parent / "metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mode", choices=["legacy", "grouped"], default="legacy")
    args = parser.parse_args()

    if args.mode == "grouped":
        evaluate_grouped(args.results, args.threshold)
    else:
        evaluate_legacy(args.results, args.threshold)
