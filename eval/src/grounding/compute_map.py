import json
import argparse
import numpy as np
import os
import re
import datetime
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, help='Path to the extracted prediction file')
parser.add_argument('--task', type=str, choices=['region-perception', 'distortion-detection', 'all'], default='all', 
                    help='Evaluation task: region-perception or distortion-detection')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching bboxes')
parser.add_argument('--output_file', type=str, help='Path to save the evaluation results', default='evaluation_results.txt')
args = parser.parse_args()

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) of two bounding boxes
    box format: [x1, y1, x2, y2]
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

def calculate_map(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP)
    gt_bboxes: format {image_id: [[box, distortion_type], ...]}
    pred_bboxes: format {image_id: [[box, distortion_type, confidence], ...]}
    
    Note: When calculating mAP, severity is ignored, only distortion type is considered
    """
    ap_list = []
    # List to store detailed results
    result_details = []
    
    # Get all distortion types (from GT and predictions)
    gt_distortion_types = set()
    pred_distortion_types = set()
    
    for boxes in gt_bboxes.values():
        for _, distortion_type in boxes:
            if distortion_type != "No distortion":
                gt_distortion_types.add(distortion_type)
    
    for boxes in pred_bboxes.values():
        for _, distortion_type, _ in boxes:
            if distortion_type != "No distortion":
                pred_distortion_types.add(distortion_type)
    
    # Merge all distortion types
    all_distortion_types = gt_distortion_types.union(pred_distortion_types)
    
    # Print statistics
    print(f"Distortion types in GT: {sorted(list(gt_distortion_types))}")
    print(f"Distortion types in predictions: {sorted(list(pred_distortion_types))}")
    print(f"Total {len(all_distortion_types)} distortion types to evaluate")
    
    result_details.append(f"Distortion types in GT: {sorted(list(gt_distortion_types))}")
    result_details.append(f"Distortion types in predictions: {sorted(list(pred_distortion_types))}")
    result_details.append(f"Total {len(all_distortion_types)} distortion types to evaluate")
    
    # Calculate AP for each distortion type
    for distortion_type in all_distortion_types:
        # Collect all boxes of current type
        gt_boxes_for_type = []
        gt_matched = []
        pred_boxes_for_type = []
        
        for img_id in gt_bboxes:
            for box, d_type in gt_bboxes[img_id]:
                if d_type == distortion_type:
                    gt_boxes_for_type.append((img_id, box))
                    gt_matched.append(False)
        
        for img_id in pred_bboxes:
            for box, d_type, confidence in pred_bboxes[img_id]:
                if d_type == distortion_type:
                    pred_boxes_for_type.append((img_id, box, confidence))
        
        # If GT doesn't have this type but predictions do, AP is 0 (all predictions are false)
        if not gt_boxes_for_type:
            message = f"Type '{distortion_type}' does not exist in GT, but has {len(pred_boxes_for_type)} boxes in predictions, AP=0.0"
            print(message)
            result_details.append(message)
            ap_list.append(0.0)
            continue
        
        # If predictions don't have this type but GT does, AP is 0 (complete recall failure)
        if not pred_boxes_for_type:
            message = f"Type '{distortion_type}' does not exist in predictions, but has {len(gt_boxes_for_type)} boxes in GT, AP=0.0"
            print(message)
            result_details.append(message)
            ap_list.append(0.0)
            continue
        
        # Sort predictions by confidence in descending order (all confidences are 1.0 here)
        pred_boxes_for_type.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate precision and recall
        tp = np.zeros(len(pred_boxes_for_type))
        fp = np.zeros(len(pred_boxes_for_type))
        
        for i, (img_id, pred_box, _) in enumerate(pred_boxes_for_type):
            max_iou = 0
            max_idx = -1
            
            # Find GT box with maximum IoU
            for j, (gt_img_id, gt_box) in enumerate(gt_boxes_for_type):
                if gt_img_id == img_id and not gt_matched[j]:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > max_iou and iou >= iou_threshold:
                        max_iou = iou
                        max_idx = j
            
            # If matched GT box found, it's a TP, otherwise it's a FP
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
        
        # Calculate AP
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            prec_at_rec = precision[recall >= r]
            if len(prec_at_rec) > 0:
                ap += np.max(prec_at_rec) / 11
        
        ap_list.append(ap)
        message = f"Type '{distortion_type}' - GT: {len(gt_boxes_for_type)} boxes, Predictions: {len(pred_boxes_for_type)} boxes, AP: {ap:.4f}"
        print(message)
        result_details.append(message)
    
    # Calculate mAP
    if ap_list:
        mAP = np.mean(ap_list)
    else:
        mAP = 0.0
    
    return mAP, result_details

def process_region_perception(pred_file, gt_file, result_lines):
    """Process evaluation for region-perception task"""
    # Load prediction and ground truth data
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    # Extract bounding box data
    gt_bboxes = defaultdict(list)  # Format: {image_id: [[box, distortion_type], ...]}
    pred_bboxes = defaultdict(list)  # Format: {image_id: [[box, distortion_type, confidence], ...]}
    
    # Process ground truth data - directly read from distortions field
    for item in gt_data:
        image_name = item['image']
        
        # Check if distortions field exists and process
        if 'distortions' in item:
            for distortion in item['distortions']:
                distortion_type = distortion.get('distortion')
                # Skip no distortion items
                if distortion_type == "No distortion":
                    continue
                
                coords = distortion.get('coordinates')
                if coords and len(coords) == 4:
                    gt_bboxes[image_name].append([coords, distortion_type])
        # If no distortions field but has answer field, try parsing from answer
        elif 'answer' in item and "No distortion affects this region" not in item['answer']:
            answer = item['answer']
            # Extract distortion type
            distortion_type = None
            if ':' in answer:
                distortion_type = answer.split(':')[0].strip()
            
            # Extract bounding box
            import re
            pattern = r'<\|box_start\|>\s*\((\d+),(\d+)\),\((\d+),(\d+)\)\s*<\|box_end\|>'
            matches = re.findall(pattern, answer)
            
            for match in matches:
                if len(match) == 4:
                    x1, y1, x2, y2 = map(int, match)
                    bbox = [x1, y1, x2, y2]
                    gt_bboxes[image_name].append([bbox, distortion_type])
    
    # Process prediction data - directly read from distortions field
    for item in pred_data:
        # Only process region-perception type questions
        if 'question_type' not in item or item['question_type'] == 'region-perception':
            image_name = item['image']
            
            # Find distortions field
            distortions = item.get('distortions', [])
            
            for distortion in distortions:
                distortion_type = distortion.get('distortion')
                # Ignore "No distortion" type
                if distortion_type == "No distortion":
                    continue
                
                # Extract coordinates
                coords = distortion.get('coordinates')
                if not coords:
                    # Check alternative field names
                    coords = distortion.get('pred_coords') or distortion.get('coords')
                
                if coords and len(coords) == 4:
                    # Set confidence to 1.0 for all predictions
                    pred_bboxes[image_name].append([coords, distortion_type, 1.0])
    
    # Count and print dataset statistics
    gt_type_count = Counter()
    pred_type_count = Counter()
    
    for boxes in gt_bboxes.values():
        for _, d_type in boxes:
            gt_type_count[d_type] += 1
    
    for boxes in pred_bboxes.values():
        for _, d_type, _ in boxes:
            pred_type_count[d_type] += 1
    
    print("\nDataset Statistics:")
    print(f"Total boxes in GT: {sum(gt_type_count.values())}")
    print(f"Total boxes in predictions: {sum(pred_type_count.values())}")
    print("\nDistortion type counts in GT:")
    
    result_lines.append("\nDataset Statistics:")
    result_lines.append(f"Total boxes in GT: {sum(gt_type_count.values())}")
    result_lines.append(f"Total boxes in predictions: {sum(pred_type_count.values())}")
    result_lines.append("\nDistortion type counts in GT:")
    
    for d_type, count in sorted(gt_type_count.items()):
        print(f"  {d_type}: {count}")
        result_lines.append(f"  {d_type}: {count}")
        
    print("\nDistortion type counts in predictions:")
    result_lines.append("\nDistortion type counts in predictions:")
    
    for d_type, count in sorted(pred_type_count.items()):
        print(f"  {d_type}: {count}")
        result_lines.append(f"  {d_type}: {count}")
        
    print()
    result_lines.append("")
    
    # Calculate mAP
    mAP, ap_details = calculate_map(gt_bboxes, pred_bboxes, args.iou_threshold)
    result_lines.extend(ap_details)
    
    print(f"\nRegion-Perception Task mAP@{args.iou_threshold}: {mAP:.4f}")
    result_lines.append(f"\nRegion-Perception Task mAP@{args.iou_threshold}: {mAP:.4f}")
    
    return mAP

def process_distortion_detection(pred_file, gt_file, result_lines):
    """Process evaluation for distortion-detection task"""
    # Load prediction and ground truth data
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    # Extract bounding box data
    gt_bboxes = defaultdict(list)  # Format: {image_id: [[box, distortion_type], ...]}
    pred_bboxes = defaultdict(list)  # Format: {image_id: [[box, distortion_type, confidence], ...]}
    
    # Process ground truth data - directly read from distortions field
    for item in gt_data:
        image_name = item['image']
        
        # Extract all distortions and their bounding boxes
        for distortion in item.get('distortions', []):
            distortion_type = distortion.get('distortion')
            coords = distortion.get('coordinates')
            
            if coords and len(coords) == 4:
                gt_bboxes[image_name].append([coords, distortion_type])
    
    # Process prediction data - directly read from distortions field
    for item in pred_data:
        # Only process distortion-detection type questions
        if 'question_type' in item and item['question_type'] == 'distortion-detection':
            image_name = item['image']
            
            # Find distortions field
            distortions = item.get('distortions', [])
            
            for distortion in distortions:
                distortion_type = distortion.get('distortion')
                # Try to get coordinates, check different possible field names
                coords = distortion.get('coordinates')
                if not coords:
                    coords = distortion.get('pred_coords') or distortion.get('coords')
                
                if coords and len(coords) == 4:
                    # Set confidence to 1.0 for all predictions
                    pred_bboxes[image_name].append([coords, distortion_type, 1.0])
    
    # Count and print dataset statistics
    gt_type_count = Counter()
    pred_type_count = Counter()
    
    for boxes in gt_bboxes.values():
        for _, d_type in boxes:
            gt_type_count[d_type] += 1
    
    for boxes in pred_bboxes.values():
        for _, d_type, _ in boxes:
            pred_type_count[d_type] += 1
    
    print("\nDataset Statistics:")
    print(f"Total boxes in GT: {sum(gt_type_count.values())}")
    print(f"Total boxes in predictions: {sum(pred_type_count.values())}")
    print("\nDistortion type counts in GT:")
    
    result_lines.append("\nDataset Statistics:")
    result_lines.append(f"Total boxes in GT: {sum(gt_type_count.values())}")
    result_lines.append(f"Total boxes in predictions: {sum(pred_type_count.values())}")
    result_lines.append("\nDistortion type counts in GT:")
    
    for d_type, count in sorted(gt_type_count.items()):
        print(f"  {d_type}: {count}")
        result_lines.append(f"  {d_type}: {count}")
    
    print("\nDistortion type counts in predictions:")
    result_lines.append("\nDistortion type counts in predictions:")
    
    for d_type, count in sorted(pred_type_count.items()):
        print(f"  {d_type}: {count}")
        result_lines.append(f"  {d_type}: {count}")
    
    print()
    result_lines.append("")
    
    # Calculate mAP
    mAP, ap_details = calculate_map(gt_bboxes, pred_bboxes, args.iou_threshold)
    result_lines.extend(ap_details)
    
    print(f"\nDistortion-Detection Task mAP@{args.iou_threshold}: {mAP:.4f}")
    result_lines.append(f"\nDistortion-Detection Task mAP@{args.iou_threshold}: {mAP:.4f}")
    
    return mAP

def main():
    # Create a list to collect all results
    all_results = []
    
    # Add evaluation time and parameter information
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_results.append(f"Evaluation Time: {current_time}")
    all_results.append(f"Prediction File: {args.pred_file}")
    all_results.append(f"IoU Threshold: {args.iou_threshold}")
    all_results.append(f"Task Type: {args.task}")
    all_results.append("-" * 60)
    
    if args.task == 'region-perception':
        gt_file = '../ViDA-MIPI-dataset/val/ViDA-Bench/ground_truth/reg-p_bench.json'
        all_results.append("\n===== Region-Perception Task Evaluation =====")
        process_region_perception(args.pred_file, gt_file, all_results)
    elif args.task == 'distortion-detection':
        metadata_file = '../ViDA-MIPI-dataset/val/ViDA-Bench/ground_truth/bench_metadata.json'
        all_results.append("\n===== Distortion-Detection Task Evaluation =====")
        process_distortion_detection(args.pred_file, metadata_file, all_results)
    else:
        gt_file = '../ViDA-MIPI-dataset/val/ViDA-Bench/ground_truth/reg-p_bench.json'
        metadata_file = '../ViDA-MIPI-dataset/val/ViDA-Bench/ground_truth/bench_metadata.json'
        
        all_results.append("\n===== Region-Perception Task Evaluation =====")
        rp_map = process_region_perception(args.pred_file, gt_file, all_results)
        
        all_results.append("\n" + "=" * 60 + "\n")
        
        all_results.append("\n===== Distortion-Detection Task Evaluation =====")
        dd_map = process_distortion_detection(args.pred_file, metadata_file, all_results)
        
        # Add average mAP of two tasks
        avg_map = (rp_map + dd_map) / 2
        all_results.append("\n" + "=" * 60)
        all_results.append(f"\nAverage mAP@{args.iou_threshold} of two tasks: {avg_map:.4f}")
    
    # Save results to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_results))
    
    print(f"\nEvaluation results saved to: {args.output_file}")

if __name__ == '__main__':
    main() 