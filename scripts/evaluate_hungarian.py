"""
Evaluate grounding predictions using Hungarian matching.
Handles multi-instance predictions by optimally matching predicted boxes to GT boxes.
"""
import json
import re
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def parse_all_bboxes(response):
    """Extract ALL bbox_2d from response."""
    return [[int(m.group(i)) for i in range(1, 5)]
            for m in re.finditer(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response)]


def hungarian_match_iou(gt_boxes, pred_boxes):
    """Compute optimal matching between GT and predicted boxes using Hungarian algorithm.
    Returns per-match IoUs and counts of unmatched GT/pred boxes."""
    if not gt_boxes or not pred_boxes:
        return [], len(gt_boxes), len(pred_boxes)

    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    cost_matrix = np.zeros((n_gt, n_pred))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            cost_matrix[i, j] = -calculate_iou(gt, pred)  # negative for minimization

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_ious = [-cost_matrix[r, c] for r, c in zip(row_ind, col_ind)]
    unmatched_gt = n_gt - len(row_ind)
    unmatched_pred = n_pred - len(row_ind)

    return matched_ious, unmatched_gt, unmatched_pred


def evaluate(predictions_file, threshold=0.5):
    with open(predictions_file) as f:
        predictions = json.load(f)

    all_ious = []
    total_gt = 0
    total_pred = 0
    total_matched = 0
    correct = 0
    failed_parse = 0

    for pred in predictions:
        gt_boxes = pred.get("ground_truth_boxes") or ([pred["ground_truth_box"]] if pred.get("ground_truth_box") else [])
        pred_boxes = pred.get("predicted_boxes") or ([pred["predicted_box"]] if pred.get("predicted_box") else [])

        gt_boxes = [b for b in gt_boxes if b is not None]
        pred_boxes = [b for b in pred_boxes if b is not None]

        if not pred_boxes:
            failed_parse += 1

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        matched_ious, unmatched_gt, unmatched_pred = hungarian_match_iou(gt_boxes, pred_boxes)
        total_matched += len(matched_ious)
        all_ious.extend(matched_ious)
        # Unmatched GTs count as IoU=0
        all_ious.extend([0.0] * unmatched_gt)

        correct += sum(1 for iou in matched_ious if iou >= threshold)

    mean_iou = np.mean(all_ious) if all_ious else 0
    recall = correct / total_gt if total_gt else 0
    precision = correct / total_pred if total_pred else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Total samples: {len(predictions)}")
    print(f"Total GT boxes: {total_gt}")
    print(f"Total predicted boxes: {total_pred}")
    print(f"Matched pairs: {total_matched}")
    print(f"Failed to parse: {failed_parse}")
    print(f"Mean IoU (matched + unmatched GT): {mean_iou:.4f}")
    print(f"Recall (IoU >= {threshold}): {recall:.4f} ({correct}/{total_gt})")
    print(f"Precision (IoU >= {threshold}): {precision:.4f} ({correct}/{total_pred})")
    print(f"F1 (IoU >= {threshold}): {f1:.4f}")

    metrics = {
        "total_samples": len(predictions),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "matched": total_matched,
        "failed_parse": failed_parse,
        "mean_iou": float(mean_iou),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "threshold": threshold,
    }

    output_file = Path(predictions_file).parent / "metrics_hungarian.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.results, args.threshold)
