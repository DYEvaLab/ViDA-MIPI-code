#! /bin/bash
python src/perception/qwen2vl.py \
    --model_path /path/to/your/local/model \
    --save_path example_result/perception/qwn2vl_vida-mipi.json

python src/perception/eval.py \
    --pred_file example_result/perception/qwn2vl_vida-mipi.json 