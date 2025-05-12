from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re
import json
import argparse
import os
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/path/to/your/local/model', help='path to the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
parser.add_argument('--question_type', type=str, default='all', choices=['all', 'reg-p', 'dist-d'], help='question type')
parser.add_argument('--image_dir', type=str, default='../ViDA-MIPI-dataset/val/images', help='path to the image directory')
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
# default: Load the model on the available device(s)
model_path = args.model_path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map=args.device,
    attn_implementation="flash_attention_2"
)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
processor = AutoProcessor.from_pretrained(args.model_path)
def process_question(question_type, data_path="../ViDA-MIPI-dataset/val/ViDA-Bench/release/benchmark.json", reg_p_path="../ViDA-MIPI-dataset/val/ViDA-Bench/release/reg-p_ques.json"):
    if question_type == 'reg-p':
        with open(reg_p_path, "r") as f:
            reg_p_data = json.load(f)
        for item in reg_p_data:
            item["question_type"] = "region-perception"
        return reg_p_data
    elif question_type == 'dist-d':
        with open(data_path, "r") as f:
            data = json.load(f)
        question = 'Please provide the bounding box coordinates of all distortions in the image.'
        for item in data:
            item["question_type"] = "distortion-detection"
            item["question"] = question
        return data
    elif question_type == 'all':
        reg_p_data = process_question('reg-p')
        dist_d_data = process_question('dist-d')
        return reg_p_data + dist_d_data

save_data = []
processed_data = process_question(args.question_type)
for data in tqdm(processed_data, total=len(processed_data)):
    type = data['question_type']
    print(type)
    sys = "You are an expert in image quality assessment."
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": os.path.join(args.image_dir, data["image"])},
            {"type": "text", "text": data["question"]}
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", f"<|im_start|>system\n{sys}<|im_end|>")
    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device).to(model.dtype)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    raw_output_text = output_text[0]
    print(raw_output_text)
    data['pred_ans'] = raw_output_text
    save_data.append(data)
    # Save the predicted answers to a file
    with open(args.save_path, 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)