import json
import argparse
import re
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, help='Path to the prediction file')
parser.add_argument('--save_dir', type=str, default='extracted_bboxes', help='Directory to save extracted bboxes')
parser.add_argument('--output_file', type=str, default='result.json', help='Output filename')
args = parser.parse_args()

def extract_bbox(text):
    """
    Extract bounding box coordinates from text
    Format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    
    Returns format: [x1, y1, x2, y2]
    """
    pattern = r'<\|box_start\|>\s*\((\d+),(\d+)\),\((\d+),(\d+)\)\s*<\|box_end\|>'
    matches = re.findall(pattern, text)
    
    bboxes = []
    for match in matches:
        if len(match) == 4:
            x1, y1, x2, y2 = map(int, match)
            bboxes.append([x1, y1, x2, y2])
    
    return bboxes

def extract_distortion_type(text):
    """
    Extract distortion type from text
    """
    # Look for distortion type, usually before colon or at the beginning
    if "No distortion" in text:
        return "No distortion"
    
    patterns = [
        r'^([^:]+):', 
        r'<\|object_ref_start\|>([^<]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return "Unknown"

def extract_severity(text):
    """
    Extract distortion severity from text
    """
    severity_patterns = [
        r'(Minor|Moderate|Severe)',
        r'<\|object_ref_start\|>(Minor|Moderate|Severe)'
    ]
    
    for pattern in severity_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return "Unknown"

def save_result(data, file_path):
    """
    Save processed data to file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}, skipping...")
        return
    
    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved result to {file_path}")

def process_predictions(pred_file):
    """Process all prediction data to extract bounding boxes"""
    # Load prediction data
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    # Copy data for adding distortions field
    result_data = []
    
    # Process prediction data
    for item in pred_data:
        # Create new item for results
        result_item = item.copy()
        result_item['distortions'] = []
        
        # Process based on different question types
        if 'question_type' in item and item['question_type'] == 'distortion-detection':
            # Process distortion-detection type questions
            # Look for answer field if pred_ans field doesn't exist
            pred_ans = item.get('pred_ans', '')
            if not pred_ans:
                pred_ans = item.get('answer', '')
            
            # Try to extract distortion type, severity, and bounding box from text
            lines = pred_ans.split('\n')
            current_distortion = None
            current_severity = None
            bbox_count = 0
            
            for i, line in enumerate(lines):
                if ':' in line and i + 1 < len(lines) and '<|box_start|>' in lines[i + 1]:
                    current_distortion = line.split(':')[0].strip()
                
                if '<|object_ref_start|>' in line:
                    severity_match = re.search(r'<\|object_ref_start\|>(Minor|Moderate|Severe)', line)
                    if severity_match:
                        current_severity = severity_match.group(1)
                
                if '<|box_start|>' in line:
                    bbox = extract_bbox(line)
                    if bbox and current_distortion:
                        for box in bbox:
                            bbox_count += 1
                            result_item['distortions'].append({
                                "id": f"bbox {bbox_count}",
                                "distortion": current_distortion,
                                "severity": current_severity or "Unknown",
                                "coordinates": box
                            })
        else:
            # Default processing as region-perception type question
            question = item.get('question', '')
            pred_ans = item.get('pred_ans', '')
            if not pred_ans:
                pred_ans = item.get('answer', '')
            
            # Extract bounding box from question
            question_bbox = extract_bbox(question)
            if question_bbox:
                # If prediction is "no distortion", distortions remains an empty list (already initialized as [])
                if "No distortion affects this region" not in pred_ans:
                    # Extract predicted bounding boxes and distortion type
                    pred_answer_bboxes = extract_bbox(pred_ans)
                    pred_distortion_type = extract_distortion_type(pred_ans)
                    pred_severity = extract_severity(pred_ans)
                    
                    if pred_answer_bboxes:
                        for i, bbox in enumerate(pred_answer_bboxes):
                            result_item['distortions'].append({
                                "id": f"bbox {i+1}",
                                "distortion": pred_distortion_type,
                                "severity": pred_severity,
                                "coordinates": bbox
                            })
        
        result_data.append(result_item)
    
    return result_data

def main():
    # Ensure save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Process all prediction data
    result_data = process_predictions(args.pred_file)
    
    # Save extracted results to a single file
    save_path = os.path.join(args.save_dir, args.output_file)
    save_result(result_data, save_path)
    print(f"All results saved to {save_path}")

if __name__ == '__main__':
    main() 