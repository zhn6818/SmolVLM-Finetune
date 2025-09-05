# SmolVLM微调框架

这个仓库包含了使用HuggingFace训练[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)的脚本。

## 🚀 快速开始

### 数据准备
1. **单图数据**：将图片和对话数据按LLaVA格式整理
2. **多图数据**：支持多张图片的对话训练
3. **视频数据**：将视频作为图像序列进行训练

### 训练方式
- **全量微调**：`bash scripts/finetune.sh`
- **LoRA微调**：`bash scripts/finetune_lora.sh`  
- **视频训练**：`bash scripts/finetune_video.sh`

## 📊 数据格式示例

### 单图数据格式
```json
[
  {
    "id": "sample_001",
    "image": "image1.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n请描述这张图片中的内容"
      },
      {
        "from": "gpt",
        "value": "这张图片显示了一个美丽的风景"
      }
    ]
  }
]
```

### 多图数据格式
```json
[
  {
    "id": "sample_002", 
    "image": ["image1.jpg", "image2.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\n比较这两张图片的差异"
      },
      {
        "from": "gpt",
        "value": "第一张图片是白天，第二张是夜晚"
      }
    ]
  }
]
```

### 视频数据格式
```json
[
  {
    "id": "video_001",
    "video": "video1.mp4",
    "conversations": [
      {
        "from": "human", 
        "value": "<video>\n这个视频中发生了什么？"
      },
      {
        "from": "gpt",
        "value": "视频中一个人在走路"
      }
    ]
  }
]
```

## ⚙️ 关键参数说明

- `--data_path`: 训练数据JSON文件路径
- `--image_folder`: 图片/视频文件夹路径
- `--learning_rate`: 语言模型学习率（建议1e-5）
- `--vision_lr`: 视觉模型学习率（建议2e-6，比语言模型小5-10倍）
- `--lora_rank`: LoRA秩（建议64）
- `--per_device_train_batch_size`: 每GPU批次大小
- `--gradient_accumulation_steps`: 梯度累积步数

## 🔧 环境配置

```bash
# 创建conda环境
conda env create -f environment.yaml
conda activate train

# 安装额外依赖
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

---

# Fine-tuning SmolVLM

This repository contains a script for training [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) with only using HuggingFace.

## Other projects

**[[Phi3-Vision Finetuning]](https://github.com/2U1/Phi3-Vision-Finetune)**<br>
**[[Qwen2-VL Finetuning]](https://github.com/2U1/Qwen2-VL-Finetune)**<br>
**[[Llama3.2-Vision Finetuning]](https://github.com/2U1/Llama3.2-Vision-Ft)**<br>
**[[Molmo Finetune]](https://github.com/2U1/Molmo-Finetune)**<br>
**[[Pixtral Finetune]](https://github.com/2U1/Pixtral-Finetune)**<br>
**[[Gemma3 Finetune]](https://github.com/2U1/Gemma3-Finetune)**

## Update

- [2025/01/24] Add option for using DoRA.
- [2025/01/24] Fixed error in LoRA.
- [2025/01/24] 🔥Supports mixed-modality data.

## Table of Contents

- [Fine-tuning SmolVLM](#fine-tuning-smolvlm)
  - [Other projects](#other-projects)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Docker](#docker)
  - [Installation](#installation)
    - [Environments](#environments)
    - [Using `requirements.txt`](#using-requirementstxt)
    - [Using `environment.yaml`](#using-environmentyaml)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
    - [Train with video dataset](#train-with-video-dataset)
      - [Merge LoRA Weights](#merge-lora-weights)
      - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- LoRA/QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image and video training

## Docker

To simplfy the setting process for training, you could use the provided pre-build environments.<br>
The settings are done in the conda env named `train`.<br><br>
You could find more information about the image [here](https://hub.docker.com/repository/docker/john119/vlm/general).

```
docker pull john119/vlm
docker run --gpus all -it -v /host/path:/docker/path --name vlm --ipc=host john119/vlm /bin/bash
```

## Installation

### Environments

- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.4

Install the required packages using `environment.yaml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

**Note:** You should install flash-attn after installing the other packages.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

**Note:** SmolVLM uses a video as a sequential of images.

</details>

## Training

**Note:** With the mixed-dataset (e.g. some data in a batch have images while some don't) It only supports with zero2.

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the SmolVLM model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--tune_connector` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--connector_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--min_pixels` (int): Option for minimum input tokens.
- `--max_pixles` (int): OPtion for maximum maxmimum tokens.
- `--lora_enable` (bool): Option for enabling LoRA (default: False)
- `--vision_lora` (bool): Option for including vision_tower to the LoRA module. The `lora_enable` should be `True` to use this option. (default: False)
- `--use_dora` (bool): Option for using DoRA instead of LoRA. The `lora_enable` should be `True` to use this option. (default: False)
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 16).
- `--lora_alpha` (int): LoRA alpha (default: 16).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

### Train with video dataset

You can train the model using a video dataset. However, SmolVLm processes videos as a sequence of images, so you’ll need to select specific frames and treat them as multiple images for training. You can set LoRA configs and use for LoRA too.

```bash
bash scripts/finetune_video.sh
```

**Note:** When training with video, it just as multi-image so you should adjust the `max_pixels` for maximum resolution and `fps` based on the available VRAM.

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json). However, using zero3 is preferred.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

#### Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## TODO

- [ ] Add feature for controlling image size.
- [ ] Add support smolvlm2.
- [ ] Add DPO Training.
- [x] Handle interleaved dataset.
- [x] Hadnle mixed-modality dataset.

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{SmolVLM-Finetuning,
  author = {Yuwon Lee},
  title = {SmolmVLM-Finetune},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/2U1/SmolVLM-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct): Awesome pretrained MLLM based on SmolLM2.
