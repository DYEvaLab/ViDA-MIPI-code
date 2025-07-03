#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import argparse
import os
import re

# 检查Python版本
if sys.version_info < (3, 6):
    print("错误：此脚本需要Python 3.6或更高版本运行。", file=sys.stderr)
    sys.exit(1)

def extract_distortions_from_text(text):
    """从预测文本中稳健地提取失真、严重性和边界框。"""
    distortions = []
    # 正则表达式，用于匹配以冒号结尾的失真类型，并捕获之后直到下一个失真类型或字符串末尾的所有内容。
    # 使用 re.MULTILINE 和非贪婪匹配来正确处理块。
    pattern = re.compile(r'(^[\w\s\-]+):\n?([\s\S]*?)(?=\n^[\w\s\-]+:|$)', re.MULTILINE)

    for match in pattern.finditer(text):
        distortion_type = match.group(1).strip()
        details_block = match.group(2)

        # 在详细信息块中查找所有 severity-box 对。一个severity可能对应多个box。
        severity_pattern = re.compile(r'(<\|object_ref_start\|>([^<]+)<\|object_ref_end\|>)((?:\s*<\|box_start\|>\(\d+,\d+\),\(\d+,\d+\)<\|box_end\|>)+)')

        for severity_match in severity_pattern.finditer(details_block):
            severity_text = severity_match.group(2).lower().strip()
            boxes_block = severity_match.group(3)

            severity = "Unknown"
            if 'moderate' in severity_text:
                severity = 'Moderate'
            elif 'severe' in severity_text:
                severity = 'Severe'
            elif 'minor' in severity_text or 'slight' in severity_text:
                severity = 'Slight'

            # 提取当前严重性对应的所有box
            box_pattern = re.compile(r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>')
            for box_match in box_pattern.finditer(boxes_block):
                x1, y1, x2, y2 = map(int, box_match.groups())
                distortions.append({
                    'distortion': distortion_type,
                    'severity': severity,
                    'coordinates': [x1, y1, x2, y2]
                })
    return distortions

def process_grounding_file(data):
    """处理输入的JSON数据，提取并替换distortions字段。"""
    print(f"开始处理数据，共 {len(data)} 条记录。")
    total_records = len(data)
    
    for idx, item in enumerate(data):
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{total_records} 条记录...")
        
        pred_ans = item.get('pred_ans', '')
        pred_ans = item.get('response', '')
        if item['type'] == "ref_grounding":
            continue
        # 提取预测bboxes
        extracted_distortions = extract_distortions_from_text(pred_ans)
        # 根据用户要求，直接使用提取结果覆盖原有的distortions字段
        # item['distortions'] = extracted_distortions
        item['pred_distortions'] = extracted_distortions
        
    print(f"所有 {total_records} 条记录处理完毕。")
    return data

def main():
    parser = argparse.ArgumentParser(description='从grounding JSON文件的预测文本中提取边界框和失真信息，并更新文件。')
    parser.add_argument('--input_file', type=str, required=True, help='输入的 grounding JSON 文件路径。')
    parser.add_argument('--output_file', type=str, required=True, help='输出更新后的JSON文件路径。')
    args = parser.parse_args()
    
    print(f"开始执行脚本，输入文件: {args.input_file}")
    
    try:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"错误：输入文件不存在 -> {args.input_file}")
        
        print("正在加载JSON文件...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            grounding_data = json.load(f)
        print("文件加载完毕。")

        modified_data = process_grounding_file(grounding_data)

        print(f"正在将更新后的数据保存到: {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f, indent=2, ensure_ascii=False)
        print("数据成功保存。")

    except json.JSONDecodeError as e:
        print(f"错误：解析JSON文件失败: {e}", file=sys.stderr)
    except Exception as e:
        import traceback
        print(f"处理过程中发生未知错误: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        print("脚本执行结束。")

if __name__ == "__main__":
    main()