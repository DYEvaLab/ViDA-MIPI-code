#! /bin/bash
python src/description/qwen2vl.py 

# You may overwrite this file to adapt to your model output format
python src/description/extract_info.py 

python src/description/eval.py 