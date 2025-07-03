import json
import os
import random
import numpy as np
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict, Counter
import re
# Set input and output paths
reference_dir = os.path.join('/app/input/', 'ref')
reference_dir = 'ref-data'
prediction_dir = os.path.join('/app/input/', 'res')
prediction_dir = 'example'
score_dir = '/app/output/'
score_dir = 'example'

# Reference file paths
mcq_file = os.path.join(reference_dir, 'mcq.json')  # For Perception Accuracy
mos_file = os.path.join(reference_dir, 'bench_metadata.json')  # For Correlation Score and Distortion mAP
region_file = os.path.join(reference_dir, 'reg-p_bench.json')  # For Region mAP

# Prediction file paths
perception_file = os.path.join(prediction_dir, 'perception.json')  # Contains MCQ answers
scoring_file = os.path.join(prediction_dir, 'scoring.json')  # Contains MOS predictions
grounding_file = os.path.join(prediction_dir, 'grounding.json')  # Contains region and distortion detection
description_file = os.path.join(prediction_dir, 'description.json')  # Contains description

# Calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    # Calculate intersection of two bounding boxes
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

def verify_grounding_data(pred_data, sample_size=10):
    """
    Randomly sample and verify that the bounding boxes in 'pred_ans' match 'distortions'.
    """
    if len(pred_data) < sample_size:
        sample_size = len(pred_data)
    
    # Filter items that have 'pred_ans' for verification
    verifiable_items = [item for item in pred_data if 'pred_ans' in item and '<|box_start|>' in item['pred_ans']]
    if not verifiable_items:
        print("No verifiable grounding data with bounding boxes in 'pred_ans'. Skipping verification.")
        return

    if len(verifiable_items) < sample_size:
        sample_size = len(verifiable_items)

    sampled_items = random.sample(verifiable_items, sample_size)
    
    for item in sampled_items:
        image_name = item['image']
        pred_ans = item.get('pred_ans', '')
        
        # Extract boxes from pred_ans string
        try:
            import re
            pred_ans_boxes = re.findall(r'<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>', pred_ans)
            pred_ans_boxes = [[int(y1), int(x1), int(y2), int(x2)] for y1, x1, y2, x2 in pred_ans_boxes]
        except Exception as e:
            print(f"Could not parse boxes from pred_ans for {image_name}: {e}")
            continue

        # Get boxes from distortions list
        distortion_boxes = [d['coordinates'] for d in item.get('distortions', []) if 'coordinates' in d]

        # Check if the boxes match (order-insensitive)
        if sorted(pred_ans_boxes) != sorted(distortion_boxes):
            raise ValueError(f"Mismatch in grounding data for image {image_name}: \
                             'pred_ans' boxes {pred_ans_boxes} do not match 'distortions' boxes {distortion_boxes}")


def calculate_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) using the all-point interpolation method (VOC2010+).
    gt_boxes: list of GT boxes for a single class, format: [[box, image_name], ...]
    pred_boxes: list of predicted boxes for a single class, format: [[box, confidence, image_name], ...]
    """
    # Sort predictions by confidence in descending order
    pred_boxes = sorted(pred_boxes, key=lambda x: x[1], reverse=True)

    num_gt = len(gt_boxes)
    if num_gt == 0:
        return 0.0

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    # Create a dictionary to track matched GT boxes for each image
    gt_matched_map = defaultdict(lambda: np.zeros(len(gt_boxes)))
    gt_boxes_by_image = defaultdict(list)
    for i, gt in enumerate(gt_boxes):
        gt_boxes_by_image[gt[1]].append((gt[0], i)) # Store box and original index

    # Match predictions to ground truth
    for i, pred in enumerate(pred_boxes):
        pred_box, _, image_name = pred
        
        gt_in_image = gt_boxes_by_image.get(image_name, [])
        best_iou = 0
        best_gt_idx = -1

        for gt_box, original_idx in gt_in_image:
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = original_idx

        if best_iou >= iou_threshold:
            if not gt_matched_map[image_name][best_gt_idx]:
                tp[i] = 1
                gt_matched_map[image_name][best_gt_idx] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # Compute precision and recall
    fp_cumsum = np.cumsum(fp)
    tp_cumsum = np.cumsum(tp)
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # All-point interpolation
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    # Calculate area under PR curve
    recall_change_indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[recall_change_indices + 1] - recalls[recall_change_indices]) * precisions[recall_change_indices + 1])
    
    print(f"  AP: {ap:.4f}")
    return ap

# Calculate mAP (mean Average Precision)
def calculate_map(gt_bboxes_map, pred_bboxes_map, iou_threshold=0.5):
    """
    Calculate mAP using all-point interpolation.
    gt_bboxes_map: {image_name: [[box, label], ...]}
    pred_bboxes_map: {image_name: [[box, label, confidence], ...]}
    """
    all_pred_boxes_by_class = defaultdict(list)
    all_gt_boxes_by_class = defaultdict(list)

    # Aggregate predictions by class
    for image_name, bboxes in pred_bboxes_map.items():
        for box, label, conf in bboxes:
            all_pred_boxes_by_class[label].append([box, conf, image_name])

    # Aggregate ground truth by class
    for image_name, bboxes in gt_bboxes_map.items():
        for box, label in bboxes:
            all_gt_boxes_by_class[label].append([box, image_name])

    all_classes = set(all_gt_boxes_by_class.keys()).union(set(all_pred_boxes_by_class.keys()))
    if not all_classes:
        print("No classes found for mAP calculation. Returning 0.")
        return 0.0
    if not all_gt_boxes_by_class:
        print("No ground truth boxes found. Returning 0 for mAP.")
        return 0.0
    if not all_pred_boxes_by_class:
        print("No prediction boxes found. Returning 0 for mAP.")
        return 0.0

    aps = []
    for label in sorted(list(all_classes)):
        gt_for_class = all_gt_boxes_by_class.get(label, [])
        pred_for_class = all_pred_boxes_by_class.get(label, [])
        print(f'Calculating AP for class: {label} ({len(gt_for_class)} GT, {len(pred_for_class)} preds)')
        ap = calculate_ap(gt_for_class, pred_for_class, iou_threshold)
        aps.append(ap)

    mean_ap = np.mean(aps) if aps else 0.0
    print(f'mAP: {mean_ap:.4f}')
    return mean_ap


def normalize_distortion_name(name):
    """
    Normalize distortion names for consistent comparison.
    """
    if not isinstance(name, str):
        return ""
    return name.lower().replace('_', ' ').strip()


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
    print(f"Key Distortion Accuracy: correct={correct}, total={total}, accuracy={accuracy:.4f}")
    return accuracy

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
    return accuracy

# 1. Calculate Perception Accuracy
def calculate_perception_accuracy():
    try:
        # Load data
        with open(perception_file, 'r') as f:
            pred_data = json.load(f)
        with open(mcq_file, 'r') as f:
            gt_data = json.load(f)
        
        # Ensure prediction and reference data have the same length
        assert len(pred_data) == len(gt_data), "Prediction and reference data have different lengths"
        
        # Sort to ensure matching
        pred_data = sorted(pred_data, key=lambda x: x["id"])
        gt_data = sorted(gt_data, key=lambda x: x["id"])
        
        # Calculate accuracy
        correct_count = 0
        total_count = len(gt_data)
        
        for pred_item, gt_item in zip(pred_data, gt_data):
            correct_ans = gt_item['correct_ans']
            # Ensure pred_ans is correctly processed
            pred_index = ord(pred_item['pred_ans'].strip()[0]) - ord('A')
            pred = pred_item['candidates'][pred_index]
            
            if correct_ans == pred:
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        return accuracy
    except Exception as e:
        return 0.0

# 2. Calculate Correlation Score
def calculate_correlation_score():
    try:
        # Load data
        with open(scoring_file, 'r') as f:
            pred_data = json.load(f)
        with open(mos_file, 'r') as f:
            gt_data = json.load(f)
        
        # Ensure prediction and reference data have the same length
        assert len(pred_data) == len(gt_data), "Prediction and reference data have different lengths"
        
        # Sort to ensure matching
        pred_data = sorted(pred_data, key=lambda x: x["id"])
        gt_data = sorted(gt_data, key=lambda x: x["id"])
        
        # Extract predicted and reference MOS values
        pred_mos = [item["pred_mos"] for item in pred_data]
        gt_mos = [item["mos"] for item in gt_data]
        
        # Calculate Spearman and Pearson correlation coefficients
        spearman_corr, _ = spearmanr(pred_mos, gt_mos)
        pearson_corr, _ = pearsonr(pred_mos, gt_mos)
        
        # Calculate correlation score
        correlation_score = (spearman_corr + pearson_corr) / 2
        return correlation_score
    except Exception as e:
        return 0.0

# 3. Calculate Region mAP
def calculate_region_map():
    try:
        # Load data
        with open(grounding_file, 'r') as f:
            pred_data = json.load(f)
        with open(region_file, 'r') as f:
            gt_data = json.load(f)

        # Verify grounding data before processing
        # verify_grounding_data(pred_data)
        
        # Extract bounding box data
        gt_bboxes = defaultdict(list)  # Format: {image_id: [[box, distortion_type], ...]}
        pred_bboxes = defaultdict(list)  # Format: {image_id: [[box, distortion_type, confidence], ...]}
        
        # Process GT data
        for item in gt_data:
            image_name = item['image']
            
            # Check distortions field and process
            if 'distortions' in item:
                for distortion in item['distortions']:
                    distortion_type = distortion.get('distortion')
                    # Skip no distortion items
                    if distortion_type == "No distortion":
                        continue
                    
                    coords = distortion.get('coordinates')
                    if coords and len(coords) == 4:
                        gt_bboxes[image_name].append([coords, distortion_type])
        
        # Process prediction data
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
        
        # Calculate Region mAP
        region_map = calculate_map(gt_bboxes, pred_bboxes, iou_threshold=0.5)
        return region_map
    except Exception as e:
        return 0.0

# 4. Calculate Distortion mAP
def calculate_distortion_map():
    try:
        # Load data
        with open(grounding_file, 'r') as f:
            pred_data = json.load(f)
        with open(mos_file, 'r') as f:
            gt_data = json.load(f)

        # Verify grounding data before processing
        # verify_grounding_data(pred_data)
        
        # Extract bounding box data
        gt_bboxes = defaultdict(list)  # Format: {image_id: [[box, distortion_type], ...]}
        pred_bboxes = defaultdict(list)  # Format: {image_id: [[box, distortion_type, confidence], ...]}
        
        # Process GT data
        for item in gt_data:
            image_name = item['image']
            
            # Extract all distortions and their bounding boxes
            for distortion in item.get('distortions', []):
                distortion_type = distortion.get('distortion')
                coords = distortion.get('coordinates')
                
                if coords and len(coords) == 4:
                    gt_bboxes[image_name].append([coords, distortion_type])
        
        # Process prediction data
        for item in pred_data:
            # Process both 'distortion-detection' and 'region-perception' types
            if 'question_type' in item and item['question_type'] in ['distortion-detection', 'region-perception']:
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
        
        # Calculate Distortion mAP
        distortion_map = calculate_map(gt_bboxes, pred_bboxes, iou_threshold=0.5)
        return distortion_map
    except Exception as e:
        print(f"Error in calculate_distortion_map: {e}")
        return 0.0

# 5. Calculate Description mAP
def calculate_description_map():
    try:
        # Load data
        with open(description_file, 'r') as f:
            pred_data = json.load(f)
        with open(mos_file, 'r') as f:
            gt_data = json.load(f)
        
        # Prepare mAP evaluation data
        gt_bboxes = defaultdict(list)
        pred_bboxes = defaultdict(list)
        
        # Process GT data
        for item in gt_data:
            image_name = item['image']
            for distortion in item.get('distortions', []):
                distortion_type = normalize_distortion_name(distortion.get('distortion'))
                coords = distortion.get('coordinates')
                if coords and len(coords) == 4:
                    gt_bboxes[image_name].append([coords, distortion_type])
        
        # Process prediction data
        for item in pred_data:
            image_name = item['image']
            for distortion in item.get('all_distortions', []):
                distortion_type = normalize_distortion_name(distortion.get('distortion'))
                coords = distortion.get('coordinates')
                if coords and len(coords) == 4:
                    pred_bboxes[image_name].append([coords, distortion_type, 1.0])
        
        # Calculate mAP
        description_map = calculate_map(gt_bboxes, pred_bboxes)
        return description_map
    except Exception as e:
        return 0.0

# 6. Calculate Key Distortion Accuracy
def calculate_key_distortion_accuracy():
    try:
        # Load data
        with open(description_file, 'r') as f:
            pred_data = json.load(f)
        with open(mos_file, 'r') as f:
            gt_data = json.load(f)
        
        key_accuracy = evaluate_key_distortions(pred_data, gt_data)
        return key_accuracy
    except Exception as e:
        return 0.0

# 7. Calculate Image Quality Accuracy
def calculate_image_quality_accuracy():
    try:
        # Load data
        with open(description_file, 'r') as f:
            pred_data = json.load(f)
        with open(mos_file, 'r') as f:
            gt_data = json.load(f)
        
        quality_accuracy = evaluate_image_quality(pred_data, gt_data)
        return quality_accuracy
    except Exception as e:
        return 0.0

# Calculate all metrics
perception_accuracy = calculate_perception_accuracy()
region_map = calculate_region_map()
distortion_map = calculate_distortion_map()
description_map = calculate_description_map()
key_distortion_accuracy = calculate_key_distortion_accuracy()
image_quality_accuracy = calculate_image_quality_accuracy()

# Calculate final score
final_score = perception_accuracy + region_map + distortion_map + description_map + key_distortion_accuracy + image_quality_accuracy

# Prepare scores dictionary
scores = {
    'Final Score': float(final_score),
    'Perception Accuracy': float(perception_accuracy),
    'Region mAP': float(region_map),
    'Distortion mAP': float(distortion_map),
    'Description mAP': float(description_map),
    'Key Distortion Accuracy': float(key_distortion_accuracy),
    'Image Quality Accuracy': float(image_quality_accuracy),
}

# Print final scores for verification
print("\n--- Final Scores ---")
print(json.dumps(scores, indent=2))
print("--------------------\n")

# Save scores
with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))


def verify_grounding_data(pred_data, sample_size=10):
    """
    Randomly sample and verify that the bounding boxes in 'pred_ans' match 'distortions'.
    """
    if len(pred_data) < sample_size:
        sample_size = len(pred_data)
    
    # Filter items that have 'pred_ans' for verification
    verifiable_items = [item for item in pred_data if 'pred_ans' in item and '<|box_start|>' in item['pred_ans']]
    if not verifiable_items:
        print("No verifiable grounding data with bounding boxes in 'pred_ans'. Skipping verification.")
        return

    if len(verifiable_items) < sample_size:
        sample_size = len(verifiable_items)

    sampled_items = random.sample(verifiable_items, sample_size)
    
    for item in sampled_items:
        image_name = item['image']
        pred_ans = item.get('pred_ans', '')
        
        # Extract boxes from pred_ans string
        try:
            pred_ans_boxes = re.findall(r'\<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)\|box_end\|\>', pred_ans)
            pred_ans_boxes = [[int(y1), int(x1), int(y2), int(x2)] for y1, x1, y2, x2 in pred_ans_boxes]
        except Exception as e:
            print(f"Could not parse boxes from pred_ans for {image_name}: {e}")
            continue

        # Get boxes from distortions list
        distortion_boxes = [d['coordinates'] for d in item.get('distortions', []) if 'coordinates' in d]

        # Check if the boxes match (order-insensitive)
        if sorted(pred_ans_boxes) != sorted(distortion_boxes):
            raise ValueError(f"Mismatch in grounding data for image {image_name}: \
                             'pred_ans' boxes {pred_ans_boxes} do not match 'distortions' boxes {distortion_boxes}")


def verify_grounding_data(pred_data, sample_size=10):
    """
    Randomly sample and verify that the bounding boxes in 'pred_ans' match 'distortions'.
    """
    if len(pred_data) < sample_size:
        sample_size = len(pred_data)
    
    # Filter items that have 'pred_ans' for verification
    verifiable_items = [item for item in pred_data if 'pred_ans' in item and '<|box_start|>' in item['pred_ans']]
    if not verifiable_items:
        print("No verifiable grounding data with bounding boxes in 'pred_ans'. Skipping verification.")
        return

    if len(verifiable_items) < sample_size:
        sample_size = len(verifiable_items)

    sampled_items = random.sample(verifiable_items, sample_size)
    
    for item in sampled_items:
        image_name = item['image']
        pred_ans = item.get('pred_ans', '')
        
        # Extract boxes from pred_ans string
        # Format: <|box_start|>(y1,x1),(y2,x2)<|box_end|>
        try:
            # Regex to find all boxes in the format (y1,x1),(y2,x2)
            parsed_boxes_str = re.findall(r'\<\|box_start\|\>\((\d+,\d+)\),\((\d+,\d+)\)\|box_end\|\>', pred_ans)
            parsed_boxes = []
            for start, end in parsed_boxes_str:
                y1, x1 = map(int, start.split(','))
                y2, x2 = map(int, end.split(','))
                parsed_boxes.append(sorted([y1, x1, y2, x2])) # Sort to be order-agnostic
        except Exception as e:
            print(f"Error parsing boxes from pred_ans for image {image_name}: {e}")
            continue

        # Get boxes from distortions list
        distortion_boxes = []
        for dist in item.get('distortions', []):
            coords = dist.get('coordinates')
            if coords and len(coords) == 4:
                distortion_boxes.append(sorted(coords))

        # Sort both lists of boxes to compare them regardless of order
        parsed_boxes.sort()
        distortion_boxes.sort()

        if parsed_boxes != distortion_boxes:
            error_msg = (
                f"Mismatch found in image {image_name}:\n"
                f"  Boxes from 'pred_ans': {parsed_boxes}\n"
                f"  Boxes from 'distortions': {distortion_boxes}"
            )
            raise ValueError(error_msg)
    
    print("Grounding data verification passed for sampled items.")

# Main execution block
if __name__ == "__main__":
    # --- Task-based Evaluation ---
    print("--- Starting Evaluation ---")

    # 1. Perception Score
    print("\n[1. Perception Accuracy]")
    perception_accuracy = calculate_perception_accuracy()
    print(f"Perception Accuracy: {perception_accuracy:.4f}")

    # 2. Image Quality Correlation Score (Not in final score, but useful)
    # print("\n[2. Correlation Score]")
    # correlation_score = calculate_correlation_score()
    # print(f"Correlation Score (SRCC+PLCC)/2: {correlation_score:.4f}")

    # 3. Region-based mAP
    print("\n[2. Region mAP]")
    region_map = calculate_region_map()
    print(f"Final Region mAP: {region_map:.4f}")

    # 4. Distortion-based mAP
    print("\n[3. Distortion mAP]")
    distortion_map = calculate_distortion_map()
    print(f"Final Distortion mAP: {distortion_map:.4f}")

    # 5. Description-based mAP
    print("\n[4. Description mAP]")
    description_map = calculate_description_map()
    print(f"Final Description mAP: {description_map:.4f}")

    # 6. Key Distortion Accuracy
    print("\n[5. Key Distortion Accuracy]")
    key_distortion_accuracy = calculate_key_distortion_accuracy()
    print(f"Final Key Distortion Accuracy: {key_distortion_accuracy:.4f}")

    # 7. Image Quality Accuracy
    print("\n[6. Image Quality Accuracy]")
    image_quality_accuracy = calculate_image_quality_accuracy()
    print(f"Final Image Quality Accuracy: {image_quality_accuracy:.4f}")

    # --- Final Score Calculation ---
    final_score = perception_accuracy + region_map + distortion_map + description_map + key_distortion_accuracy + image_quality_accuracy

    # --- Final Scores Summary ---
    scores = {
        "Final Score": final_score,
        "Perception Accuracy": perception_accuracy,
        "Region mAP": region_map,
        "Distortion mAP": distortion_map,
        "Description mAP": description_map,
        "Key Distortion Accuracy": key_distortion_accuracy,
        "Image Quality Accuracy": image_quality_accuracy
    }

    print("\n--- Final Scores ---")
    print(json.dumps(scores, indent=2))
    print("--------------------\n")

    # Write scores to file
    output_path = os.path.join(score_dir, 'scores.json')
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=4)