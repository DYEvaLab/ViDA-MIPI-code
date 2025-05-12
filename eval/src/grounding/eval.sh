#! /bin/bash
python src/grounding/qwen2vl.py \
    --model_path /path/to/your/local/models \
    --save_path example_result/grounding/qwen2vl_vida-mipi.json

python src/grounding/extract_bbox.py \
    --pred_file example_result/grounding/qwen2vl_vida-mipi.json \
    --save_dir example_result/grounding \
    --output_file extracted_ƒbboxes.json

python src/grounding/compute_map.py \
    --pred_file example_result/grounding/extracted_bboxes.json 
