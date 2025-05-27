#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import argparse
from collections import defaultdict

def calculate_iou(box1, box2):
    """
    Calculate IoU of two bounding boxes
    Box format: [x1, y1, x2, y2]
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def normalize_distortion_name(name):
    """Normalize distortion names for comparison"""
    if not name:
        return ""
    
    # Convert to lowercase and remove extra spaces
    name = name.lower().strip()
    
    if name == 'blocking artifact':
        name = 'blocking artifacts'
    return name

def calculate_map_for_distortions(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    """
    Calculate mAP for all distortion boxes
    gt_bboxes: {image_name: [(bbox, distortion_type), ...]}
    pred_bboxes: {image_name: [(bbox, distortion_type), ...]}
    """
    ap_list = []
    
    # Get all distortion types
    gt_distortion_types = set()
    pred_distortion_types = set()
    
    for boxes in gt_bboxes.values():
        for _, distortion_type in boxes:
            gt_distortion_types.add(normalize_distortion_name(distortion_type))
    
    for boxes in pred_bboxes.values():
        for _, distortion_type in boxes:
            pred_distortion_types.add(normalize_distortion_name(distortion_type))
    
    all_distortion_types = gt_distortion_types.union(pred_distortion_types)
    
    print(f"Distortion types in GT: {sorted(list(gt_distortion_types))}")
    print(f"Distortion types in predictions: {sorted(list(pred_distortion_types))}")
    print(f"Total {len(all_distortion_types)} distortion types to evaluate")
    
    # Calculate AP for each distortion type
    for distortion_type in all_distortion_types:
        # Collect all boxes of current type
        gt_boxes_for_type = []
        gt_matched = []
        pred_boxes_for_type = []
        
        for img_name in gt_bboxes:
            for box, d_type in gt_bboxes[img_name]:
                if normalize_distortion_name(d_type) == distortion_type:
                    gt_boxes_for_type.append((img_name, box))
                    gt_matched.append(False)
        
        for img_name in pred_bboxes:
            for box, d_type in pred_bboxes[img_name]:
                if normalize_distortion_name(d_type) == distortion_type:
                    pred_boxes_for_type.append((img_name, box, 1.0))  # Set confidence to 1.0
        
        if not gt_boxes_for_type:
            print(f"Type '{distortion_type}' does not exist in GT, but has {len(pred_boxes_for_type)} boxes in predictions, AP=0.0")
            ap_list.append(0.0)
            continue
        
        if not pred_boxes_for_type:
            print(f"Type '{distortion_type}' does not exist in predictions, but has {len(gt_boxes_for_type)} boxes in GT, AP=0.0")
            ap_list.append(0.0)
            continue
        
        # Sort by confidence (all are 1.0 here)
        pred_boxes_for_type.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate precision and recall
        tp = np.zeros(len(pred_boxes_for_type))
        fp = np.zeros(len(pred_boxes_for_type))
        
        for i, (img_name, pred_box, _) in enumerate(pred_boxes_for_type):
            max_iou = 0
            max_idx = -1
            
            # Find GT box with maximum IoU
            for j, (gt_img_name, gt_box) in enumerate(gt_boxes_for_type):
                if gt_img_name == img_name and not gt_matched[j]:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > max_iou and iou >= iou_threshold:
                        max_iou = iou
                        max_idx = j
            
            if max_idx >= 0:
                tp[i] = 1
                gt_matched[max_idx] = True
            else:
                fp[i] = 1
        
        # Calculate cumulative TP and FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / len(gt_boxes_for_type)
        
        # Calculate AP (11-point interpolation)
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            prec_at_rec = precision[recall >= r]
            if len(prec_at_rec) > 0:
                ap += np.max(prec_at_rec) / 11
        
        ap_list.append(ap)
        print(f"Type '{distortion_type}' - GT: {len(gt_boxes_for_type)} boxes, Predictions: {len(pred_boxes_for_type)} boxes, AP: {ap:.4f}")
    
    # Calculate mAP
    if ap_list:
        mAP = np.mean(ap_list)
    else:
        mAP = 0.0
    
    return mAP, ap_list

def evaluate_key_distortions(pred_data, gt_data):
    """
    Evaluate key distortion accuracy
    Correct if distortion name matches and box IoU > 0.5, otherwise incorrect
    """
    correct = 0
    total = 0
    
    # Create GT data index
    gt_dict = {}
    for item in gt_data:
        gt_dict[item['image']] = item
    
    for pred_item in pred_data:
        image_name = pred_item['image']
        pred_key_distortions = pred_item.get('key_distortions', [])
        
        if image_name not in gt_dict:
            continue
        
        gt_item = gt_dict[image_name]
        gt_distortions = gt_item.get('distortions', [])
        
        # Find distortions with highest significance_score in GT as key distortions
        if not gt_distortions:
            continue
        
        # Sort by significance_score, take those with highest score as key distortions
        # Find the highest significance_score
        max_significance = max(d.get('significance_score', 0) for d in gt_distortions)
        # Take all distortions with highest significance_score as key distortions
        gt_key_distortions = [d for d in gt_distortions if d.get('significance_score', 0) == max_significance]
        
        # For each predicted key distortion, check if it matches GT
        for pred_key in pred_key_distortions:
            total += 1
            pred_distortion = normalize_distortion_name(pred_key['distortion'])  
            pred_coords = pred_key['coordinates']
            
            # Check if it matches any GT key distortion
            matched = False
            for gt_key in gt_key_distortions:  # Check all GT key distortions
                gt_distortion = normalize_distortion_name(gt_key['distortion'])  
                gt_coords = gt_key['coordinates']
                
                # Check if distortion names match (direct comparison without normalize)
                if pred_distortion.lower().strip() == gt_distortion.lower().strip():
                    # Check if IoU > 0.5
                    iou = calculate_iou(pred_coords, gt_coords)
                    if iou > 0.5:
                        matched = True
                        break
            
            if matched:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def evaluate_image_quality(pred_data, gt_data):
    """
    Evaluate image quality accuracy
    """
    correct = 0
    total = 0
    
    # Create GT data index
    gt_dict = {}
    for item in gt_data:
        gt_dict[item['image']] = item
    
    for pred_item in pred_data:
        image_name = pred_item['image']
        pred_quality = pred_item.get('image_quality', '')
        
        if image_name not in gt_dict:
            continue
        
        gt_item = gt_dict[image_name]
        gt_quality = gt_item.get('level', '')
        
        total += 1
        if pred_quality == gt_quality:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def main():
    # Create argument parser in main function
    parser = argparse.ArgumentParser(description='ViDA-MIPI Description Task Evaluation')
    parser.add_argument('--pred_file', type=str, default='eval/example_result/description/extracted_info.json', help='Path to the extracted prediction file')
    parser.add_argument('--gt_file', type=str, default='../ViDA-MIPI-dataset/val/ViDA-Bench/ground_truth/bench_metadata.json', help='Path to the ground truth metadata file')
    parser.add_argument('--output_file', type=str, default='eval/example_result/description/results.txt', help='Output file for results')
    args = parser.parse_args()
    
    print("=== ViDA-MIPI Description Task Evaluation ===\n")
    
    # Load data
    print("Loading prediction data...")
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    print("Loading GT data...")
    with open(args.gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    print(f"Prediction data: {len(pred_data)} samples")
    print(f"GT data: {len(gt_data)} samples\n")
    
    # Prepare mAP evaluation data
    print("=== 1. All distortion boxes vs metadata mAP evaluation ===")
    gt_bboxes = defaultdict(list)
    pred_bboxes = defaultdict(list)
    
    # Process GT data
    for item in gt_data:
        image_name = item['image']
        for distortion in item.get('distortions', []):
            distortion_type = distortion.get('distortion')
            coords = distortion.get('coordinates')
            if coords and len(coords) == 4:
                gt_bboxes[image_name].append((coords, distortion_type))
    
    # Process prediction data
    for item in pred_data:
        image_name = item['image']
        for distortion in item.get('all_distortions', []):
            distortion_type = distortion.get('distortion')
            coords = distortion.get('coordinates')
            if coords and len(coords) == 4:
                pred_bboxes[image_name].append((coords, distortion_type))
    
    # Calculate mAP
    mAP, ap_list = calculate_map_for_distortions(gt_bboxes, pred_bboxes)
    print(f"\nAll distortion boxes mAP: {mAP:.4f}")
    
    # Evaluate key distortion accuracy
    print("\n=== 2. Key distortion accuracy evaluation ===")
    key_accuracy, key_correct, key_total = evaluate_key_distortions(pred_data, gt_data)
    print(f"Key distortion accuracy: {key_accuracy:.4f} ({key_correct}/{key_total})")
    
    # Evaluate image quality accuracy
    print("\n=== 3. Image quality accuracy evaluation ===")
    quality_accuracy, quality_correct, quality_total = evaluate_image_quality(pred_data, gt_data)
    print(f"Image quality accuracy: {quality_accuracy:.4f} ({quality_correct}/{quality_total})")
    
    # Save results
    results = [
        "=== ViDA-MIPI Description Task Evaluation Results ===\n",
        f"Prediction file: {args.pred_file}",
        f"GT file: {args.gt_file}",
        f"Prediction samples: {len(pred_data)}",
        f"GT samples: {len(gt_data)}\n",
        "=== 1. All distortion boxes mAP evaluation ===",
        f"mAP: {mAP:.4f}\n",
        "=== 2. Key distortion accuracy evaluation ===",
        f"Accuracy: {key_accuracy:.4f} ({key_correct}/{key_total})\n",
        "=== 3. Image quality accuracy evaluation ===",
        f"Accuracy: {quality_accuracy:.4f} ({quality_correct}/{quality_total})\n",
        "=== Summary ===",
        f"Distortion detection mAP: {mAP:.4f}",
        f"Key distortion accuracy: {key_accuracy:.4f}",
        f"Image quality accuracy: {quality_accuracy:.4f}"
    ]
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    
    print(f"\nEvaluation results saved to: {args.output_file}")
    print("\n=== Summary ===")
    print(f"Distortion detection mAP: {mAP:.4f}")
    print(f"Key distortion accuracy: {key_accuracy:.4f}")
    print(f"Image quality accuracy: {quality_accuracy:.4f}")

if __name__ == '__main__':
    main() 