# Code for Detailed Image Quality Assessment Track in MIPI Challenge

This repository contains both training code and evalutation code for [Detailed Image Quality Assessment Track in MIPI Challenge](https://www.codabench.org/competitions/8156/?secret_key=2a29d769-91c5-47d6-80c8-5f1279b7495c). This track decomposes the traditional IQA task from a score fitting task into three subtasks: fine-grained quality grounding, detailed quality perception, and reasoning quality description. We strive to empower existing vision-language foundation models to function as all-in-one solutions, enabling a single model to accomplish all tasks and excel in detailed  IQA. Training code contains a script for fine-tuning [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) with our [Visual Distortion Assessment Dataset](https://huggingface.co/datasets/DY-Evalab/ViDA-MIPI-dataset). For evaluation convenience, We provide a fine-tuned baseline model [Qwen2-VL-ViDA](https://huggingface.co/jasonliaonk21/Qwen2-VL-ViDA) using the [training script](./train/finetune_ViDA_lora.sh). Evaluation code contains evaluating Qwen2-VL-ViDA from image quality scoring, image quality perception and image quality grounding.


## Update
- [2025/05/12] 🔥Initial Commit

## Table of Contents

- [Code for Detailed Image Quality Assessment Track in MIPI Challenge]()
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Using `requirements.txt`](#using-requirementstxt)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [TODO](#todo)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)


## Installation

Install the required packages using `requirements.txt`.

### Using `requirements.txt`

```bash
conda env create -n vida-mipi python==3.12
conda activate vida-mipi
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.

## Dataset Preparation

- Download the [ViDA-MIPI-dataset](https://huggingface.co/datasets/DY-Evalab/ViDA-MIPI-dataset) and unzip the images in train/val folders. 
- Arrange the folders as follows:

```
|-- ViDA-MIPI-code
|-- ViDA-MIPI-dataset
  |-- train
    |-- images/*.jpg
    |-- ViDA-qwen2
      |-- ViDA-D(reasoning quality description)
        |-- assess.jsonl
        |-- brief_assess.jsonl
        |-- dist_assess.jsonl
      |-- ViDA-G(fine-grained quality grounding)
        |-- dist_detect.jsonl
        |-- ref-grounding.jsonl
        |-- reg-perception.jsonl
      |-- ViDA-P(detailed quality perception)
        |-- mcq.jsonl
        |-- qa.jsonl
        |-- score.jsonl
      |-- combined.json
      |-- jsonl_stats.txt
    |-- train_metadata.json
  |-- val
    |-- images/*.jpg
    |-- ViDA-Bench
      |-- images/*.jpg
      |-- release
```
- Training dataset contains 11,848 images and Valiadation dataset contains 477 images.
**Note:** 
- The bounding box coordinates in all files have been normalized to (0, 1000), which is consistent with the output format of Qwen2-VL.
- We use `combined.json` to fine-tune Qwen2-VL. This JSON file is the combination of all JSONL files. It contains 534K samples and is formatted according to the LLaVA specification, where each entry contains information about conversations and images. 
- The JSONL files in the ViDA-qwen2 folders are generated from the data in `train_metadata.json` using GPT-4o or converted via manually defined formats. Data statistics are in `jsonl_stats.txt`.

## Training
### Finetune with LoRA
```bash
cd train
bash finetune_ViDA_lora.sh
```
- Our training code is a modification from [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune). Please refer to this repository for more training methods and experiment settings. We also recommend [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning. 
- We use **8 H20 GPUs** to finetune Qwen2-VL-7B-Instruct. Training with the full dataset takes approximately 8 days, but by the 4th day, the model's performance on the validation set had already approached saturation.

## Evaluation
```bash
cd eval
# evaluate image quality scoring 
bash src/score/eval.sh
# evaluate image quality perception 
bash src/perception/eval.sh
# evaluate image quality grounding
bash src/grounding/eval.sh
```
- We provide the inference outputs of Qwen2-VL-ViDA in [example_result](./eval/example_result).
- Since the ground truth data for the validation dataset is not provided, the three scripts mentioned above can only serve as references and cannot run successfully.


## TODO

- [x] Release the dataset and code 
- [ ] Release the link of MIPI Challenge Track
- [ ] Release the paper


## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star:. We will release the paper for citation as soon as possible.

## Acknowledgement

This project is based on

- [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune): An open-source implementaion for fine-tuning Qwen2-VL and Qwen2.5-VL series by Alibaba Cloud.
- [Q-Align](https://github.com/Q-Future/Q-Align): All-in-one Foundation Model for visual scoring. Can efficiently fine-tune to downstream datasets.
- [Q-Bench](https://github.com/Q-Future/Q-Bench): A benchmark for multi-modality LLMs (MLLMs) on low-level vision and visual quality assessment.
