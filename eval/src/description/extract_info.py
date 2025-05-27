import json
import argparse
import re
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, help='Path to the prediction file')
parser.add_argument('--save_dir', type=str, default='eval/example_result/description', help='Directory to save extracted info')
parser.add_argument('--output_file', type=str, default='extracted_info.json', help='Output filename')
args = parser.parse_args()

def extract_coordinates(text):
    """
    Extract coordinates from text
    Format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    
    Return format: [x1, y1, x2, y2]
    """
    pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches = re.findall(pattern, text)
    
    coordinates = []
    for match in matches:
        if len(match) == 4:
            x1, y1, x2, y2 = map(int, match)
            coordinates.append([x1, y1, x2, y2])
    
    return coordinates

def extract_distortion_info(text):
    """
    Extract distortion information from text
    Return: distortion type, severity, coordinates
    """
    distortions = []
    seen_distortions = set()  # For deduplication
    
    # Match pattern 1: "There is one <|object_ref_start|>severe overexposure<|object_ref_end|><|box_start|>(430,391),(545,495)<|box_end|>"
    pattern1 = r'There is (?:one|two|three|\d+) <\|object_ref_start\|>(\w+) ([^<]+)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches1 = re.findall(pattern1, text)
    
    for match in matches1:
        if len(match) == 6:
            severity = match[0]  # severe, moderate, minor
            distortion_type = match[1].strip()  # overexposure, low clarity, etc.
            x1, y1, x2, y2 = map(int, match[2:6])
            
            # Create unique identifier for deduplication
            distortion_key = (distortion_type, severity, x1, y1, x2, y2)
            if distortion_key not in seen_distortions:
                seen_distortions.add(distortion_key)
                distortions.append({
                    "distortion": distortion_type,
                    "severity": severity,
                    "coordinates": [x1, y1, x2, y2]
                })
    
    # Match pattern 2: Direct "<|object_ref_start|>Moderate low clarity<|object_ref_end|><|box_start|>(199,10),(922,320)<|box_end|>"
    pattern2 = r'<\|object_ref_start\|>(\w+) ([^<]+)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches2 = re.findall(pattern2, text)
    
    for match in matches2:
        if len(match) == 6:
            severity = match[0].lower()  # Moderate -> moderate
            distortion_type = match[1].strip()  # low clarity, edge aliasing effect, etc.
            x1, y1, x2, y2 = map(int, match[2:6])
            
            # Create unique identifier for deduplication
            distortion_key = (distortion_type, severity, x1, y1, x2, y2)
            if distortion_key not in seen_distortions:
                seen_distortions.add(distortion_key)
                distortions.append({
                    "distortion": distortion_type,
                    "severity": severity,
                    "coordinates": [x1, y1, x2, y2]
                })
    
    return distortions

def extract_key_distortion(text):
    """
    Extract key distortion information from text
    Extract from the first sentence of the second-to-last line after split('\n')
    Return list format to support multiple key distortions
    """
    lines = text.split('\n')
    if len(lines) < 2:
        return []
    
    # Get the second-to-last part
    second_last_line = lines[-2].strip()
    
    # Split by period, take the first sentence
    sentences = second_last_line.split('.')
    if not sentences:
        return []
    
    first_sentence = sentences[0].strip()
    
    # Extract all key distortion information
    key_distortions = []
    seen_distortions = set()  # For deduplication
    
    # Match pattern: "The <|object_ref_start|>severe overexposure<|object_ref_end|><|box_start|>(430,391),(545,495)<|box_end|>"
    pattern = r'<\|object_ref_start\|>(\w+) ([^<]+)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches = re.findall(pattern, first_sentence)
    
    for match in matches:
        if len(match) == 6:
            severity = match[0].lower()  # severe -> severe, Moderate -> moderate
            distortion_type = match[1].strip()
            x1, y1, x2, y2 = map(int, match[2:6])
            
            # Create unique identifier for deduplication
            distortion_key = (distortion_type, severity, x1, y1, x2, y2)
            if distortion_key not in seen_distortions:
                seen_distortions.add(distortion_key)
                key_distortions.append({
                    "distortion": distortion_type,
                    "severity": severity,
                    "coordinates": [x1, y1, x2, y2]
                })
    
    return key_distortions

def extract_image_quality(text):
    """
    Extract image quality rating from text
    Extract from the last part after split('\n')
    """
    lines = text.split('\n')
    if not lines:
        return "unknown"
    
    # Get the last part
    last_line = lines[-1].strip()
    
    # Extract quality rating from the last line
    quality_pattern = r'the quality of the image is (\w+)'
    match = re.search(quality_pattern, last_line, re.IGNORECASE)
    
    if match:
        quality = match.group(1).lower()
        # Normalize quality levels
        quality_mapping = {
            'poor': 'poor',
            'bad': 'bad', 
            'fair': 'fair',
            'good': 'good',
            'excellent': 'excellent'
        }
        return quality_mapping.get(quality, quality)
    
    return "unknown"

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
    
    print(f"Results saved to {file_path}")

def process_predictions(pred_file):
    """Process all prediction data to extract information"""
    # Load prediction data
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Copy data for adding extracted information
    result_data = []
    
    # Process prediction data
    for item in pred_data:
        # Create new result item
        result_item = item.copy()
        
        # Get prediction answer
        pred_ans = item.get('pred_ans', '')
        if not pred_ans:
            pred_ans = item.get('answer', '')
        
        # 1. Extract all distortion information
        all_distortions = extract_distortion_info(pred_ans)
        result_item['all_distortions'] = all_distortions
        
        # 2. Extract key distortion information
        key_distortions = extract_key_distortion(pred_ans)
        result_item['key_distortions'] = key_distortions
        
        # 3. Extract image quality
        image_quality = extract_image_quality(pred_ans)
        result_item['image_quality'] = image_quality
        
        result_data.append(result_item)
    
    return result_data

def main():
    # Ensure save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Process all prediction data
    result_data = process_predictions(args.pred_file)
    
    # Save extracted results to single file
    save_path = os.path.join(args.save_dir, args.output_file)
    save_result(result_data, save_path)
    print(f"All results saved to {save_path}")

if __name__ == '__main__':
    main()
