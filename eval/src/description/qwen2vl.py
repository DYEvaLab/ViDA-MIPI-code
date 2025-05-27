from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re
import json
import argparse
import os
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/Qwen2-VL-ViDA/', help='path to the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--save_path', type=str, default='eval/example_result/description/qwen2vl_vida-mipi.json', help='path to save the predicted answers')
parser.add_argument('--image_dir', type=str, default='../ViDA-MIPI-dataset/val/images', help='path to the image directory')
parser.add_argument('--max_new_tokens', default=2048, type=int, help='max number of new tokens to generate')
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

with open('../ViDA-MIPI-dataset/val/ViDA-Bench/release/benchmark.json', 'r') as f:
    data = json.load(f)

# default: Load the model on the available device(s)
model_path = args.model_path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map=args.device,
    attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(args.model_path)
save_data = []
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

system_prompt = "You are an expert in image quality assessment. Your task is to assess the overall quality of the provided image.\nTo assess the image quality, you should think step by step.\nFirst step, provide a brief description of the image content.\nSecond step, find distortions and analyze their impact on specific regions.\n- If there is no distortion present in the image, focus solely on what you observe in the image and describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture.\n- If distortions are present, identify the distortions and briefly analyze each occurrence of every distortion type. Explain how each distortion affects the visual appearance and perception of specific objects or regions in the image. Find each distortion with grounding, which means in the response you should directly output the bounding box coordinates after identifying the distortions. The bounding box coordinates should be in the format (x1,y1),(x2,y2), where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner. All coordinates are normalized to the range of 0 to 1000.\nThird step, analyze the overall image quality and visual perception. If distortions are present, identify the key distortions that have the most significant impact on the overall image quality. Provide detailed reasoning about how these key distortions affect the image's overall visual perception, especially regarding low-level features like sharpness, clarity, and detail. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph.\nFinal step, conclude your answer with this sentence: Thus, the quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent).\nSeparate each step with a line break."

for item in tqdm(data, total=len(data)):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": os.path.join(args.image_dir, item['image']),
            },
            {"type": "text", "text": "Please evaluate the image quality in detail and provide your reasons."},
        ],
    }
]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", f"<|im_start|>system\n{system_prompt}<|im_end|>")
    # print(text)
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
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    item['pred_ans'] = output_text[0]
    save_data.append(item)
    # Save the predicted answers to a file
    with open(args.save_path, 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)