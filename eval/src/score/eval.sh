#! /bin/bash

python src/score/qwen2vl.py \
    --model_path /path/to/your/local/model \
    --save_path example_result/score/qwn2vl_vida-mipi.json

python src/score/eval.py \
    --pred_json example_result/score/qwn2vl_vida-mipi.json 