"""Inference script for Pi models on Airbot dataset."""

import sys

sys.path.insert(0, "/home/jovyan/projects/Pi/src")

import random
import dataclasses

import numpy as np
import torch

from pi.policies import policy_config_torch
from pi.training import config as _config, data_loader as _data_loader


def set_random_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seeds(0)


config = _config.get_config("pi05_airbot")
config = dataclasses.replace(config, batch_size=4)
checkpoint_dir = "/home/jovyan/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch"

# 手动加载我们的norm_stats
data_config = config.data.create(config.assets_dirs, config.model)
manual_norm_stats = data_config.norm_stats

# 创建policy时传入我们的norm_stats
policy = policy_config_torch.create_trained_policy(config, checkpoint_dir, norm_stats=manual_norm_stats)

dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)

# 按照libero模式构建输入，使用简化的键名
sample = dataset[100]
data_input = {
    "observation/state": sample["observation.state"],
    "observation/cam_env": sample["observation.images.cam_env"],
    "observation/cam_left_wrist": sample["observation.images.cam_left_wrist"],
    "observation/cam_right_wrist": sample["observation.images.cam_right_wrist"],
    "actions": sample["action"],  # 添加action键
    "prompt": sample["task"],
}


example = data_input
noise = np.random.randn(1, 10, 32).astype(np.float32)

action_chunk = policy.infer(example, noise=noise)["actions"]
print(f"action_chunk shape: {action_chunk.shape}")
print(f"action_chunk: {action_chunk}")

import json


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj


output_data = {
    "example": {k: convert_to_serializable(v) for k, v in example.items()},
    "action_chunk": convert_to_serializable(action_chunk),
}

with open("inference_output_airbot_torch.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("example和action_chunk已保存到 inference_output_airbot_torch.json 文件中。")
