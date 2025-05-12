import json
import os
import argparse
import debugpy
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from torch import nn
from typing import List, Dict, Any

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception:
#     pass
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/path/to/your/local/model', help='path to the model')
parser.add_argument('--device', type=str, default='cuda', help='device to run the model')
parser.add_argument('--eval_file', type=str, default='../ViDA-MIPI-dataset/val/ViDA-Bench/release/benchmark.json', help='path to the evaluation file')
parser.add_argument('--image_folder', type=str, default='../ViDA-MIPI-dataset/val/images', help='path to the folder of images')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
args = parser.parse_args()

def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

class Qwen2QAlignScorer(nn.Module):
    def __init__(
        self,
        model_path,
        device,
        level=[" excellent", " good", " fair", " poor", " bad"],
    ):
        """
        Initializes the Scorer class.

        Args:
            model_path (str): The path to the pretrained model.
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda' or 'cuda:0'), if device is "cuda", device_map will be "auto", otherwise, device_map will be device.
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
        """
        super().__init__()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.level = level
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.tokenizer = self.processor.tokenizer
        print(self.processor.image_processor.max_pixels)
        self.cal_ids_ = [
            id_[0]
            for id_ in self.tokenizer(
                [" excellent", " good", " fair", " poor", " bad"]
            )["input_ids"]
        ]
        self.preferential_ids_ = [id_[0] for id_ in self.tokenizer(level)["input_ids"]]

        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(self.dtype).to(self.device)
        )

    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = "You are an expert in image quality assessment. Your task is to give a one-sentence assessment of the image quality: The quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent).",
    ):
        """
        Rates the quality of a list of input images, returning a score for each image between 1 and 5.

        Args:
            image_path (List[str]): The list of file paths for input images.

        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - A tensor containing the calculated score for each image, ranging from 1 to 5.
                - A list of dictionaries, each containing the filename and logits for the respective image.
        """
        if sys_prompt is not None:
            prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Assess the quality of this image.<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
        else:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Assess the quality of this image.<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
        prompts = [prompt] * len(image_path)
        with torch.inference_mode(): 
            output_logits = []
            cal_logits = []
            id = 0
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                inputs = self.processor(
                    images=[load_image(path)], text=[prompt], return_tensors="pt"
                ).to(self.device, self.dtype)
                logit = self.model(**inputs)["logits"]
                output_logit = (
                    logit[:, -1, self.preferential_ids_]
                    .to(self.dtype)
                    .squeeze()
                    .tolist()
                )
                cal_logit = logit[:, -1, self.cal_ids_].to(self.dtype)
                print(cal_logit)
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )

                output_logits.append(
                    {"id": id, "filename": os.path.basename(path), "logits": logits_dict}
                )
                id += 1

            cal_logits = torch.stack(cal_logits, 0).squeeze()
            pred_mos_values = (
                torch.softmax(cal_logits, -1) @ self.weight_tensor
            ).tolist()
            if isinstance(pred_mos_values, float):
                pred_mos_values = [pred_mos_values]

            for i, output in enumerate(output_logits):
                output["pred_mos"] = pred_mos_values[i]

            return output_logits


file = args.eval_file

with open(file, "r", encoding="utf-8") as file:
    data = json.load(file)
img_list = []
image_dir = args.image_folder
save_path = args.save_path
os.makedirs(os.path.dirname(save_path), exist_ok=True)
for i in range(len(data)):
    try:
        image = data[i]["image"]
    except:
        image = data[i]["filename"]
    img_list.append(os.path.join(image_dir, image))

model_path = args.model_path
levels = [
    " excellent",
    " good",
    " fair",
    " poor",
    " bad",
    " high",
    " low",
    " fine",
    " moderate",
    " decent",
    " average",
    " medium",
    " acceptable",
]
device = args.device

scorer = Qwen2QAlignScorer(model_path, device=device, level=levels)
output = scorer(img_list)
print("Saving results to", save_path)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as file:
    json.dump(output, file, ensure_ascii=False, indent=4)