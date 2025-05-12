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
parser.add_argument('--image_dir', type=str, default='../ViDA-MIPI-dataset/val/images', help='path to the image directory')
parser.add_argument('--json_file', type=str, default='../ViDA-MIPI-dataset/val/ViDA-Bench/release/mcq.json', help='path to the json file')
args = parser.parse_args()

def convert_to_mcq(data):
    # Extract relevant information from the input dictionary
    question = data["question"]
    candidates = data["candidates"]
    
    # Map the candidates to letters A, B, C, ...
    options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(candidates)])
    
    # Format the question and options in the desired output format
    formatted_output = f"{question}\n{options}\nAnswer with a single option's letter (A, B, C, ...) from the given choices directly."
    
    return formatted_output

def process_benchmark_mcq(json_file, image_dir):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    processed_data = []
    for item in raw_data:
        image_path = os.path.join(image_dir, os.path.basename(item["image"]))
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist. Skipping...")
            input()
            continue
        text = convert_to_mcq(item)
        processed_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
        processed_data.append(processed_item)
    return raw_data, processed_data

raw_data, processed_data = process_benchmark_mcq(args.json_file, args.image_dir)

if os.path.exists(args.save_path):
    print(f"File {args.save_path} already exists. Exiting...")
    exit()
else:
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
# default: Load the model on the available device(s)
model_path = args.model_path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map=args.device,
    attn_implementation="flash_attention_2"
)

print(model.hf_device_map)
print(model.dtype)  

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 4*28*28
max_pixels = 4096*28*28
processor = AutoProcessor.from_pretrained(args.model_path)

for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    data['content'][0]['min_pixels'] = min_pixels
    data['content'][0]['max_pixels'] = max_pixels
    messages = [data]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    gt["pred_ans"] = output_text[0]  # Handle cases where output_text is not as expected

    try:
        print("GT:", gt["correct_ans"])
        print("Pred:", gt["pred_ans"])
    except:
        print("Pred:", gt["pred_ans"])

    # Save the predicted answers to a file
    with open(args.save_path, 'w') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)
