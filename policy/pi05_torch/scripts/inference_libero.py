"""Inference script for Pi models on Libero dataset."""

import sys

sys.path.insert(0, "/home/jovyan/projects/Pi/src")

import random
import dataclasses

import numpy as np
import torch

from pi.policies import policy_config_torch
from pi.training import config as _config, data_loader as _data_loader


# 设置所有随机种子以确保结果可复现
def set_random_seeds(seed=0):
    # Python内置random模块
    random.seed(seed)

    # Numpy随机种子
    np.random.seed(seed)

    # PyTorch随机种子
    torch.manual_seed(seed)

    # CUDA随机种子（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况

        # 设置CUDA的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 调用函数设置随机种子
set_random_seeds(0)


config = _config.get_config("pi05_libero")
config = dataclasses.replace(config, batch_size=4)
checkpoint_dir = "/home/jovyan/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config_torch.create_trained_policy(config, checkpoint_dir)


data_config = config.data.create(config.assets_dirs, config.model)
dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
data_input = {
    "observation/state": dataset[100]["state"],
    "observation/image": dataset[100]["image"],
    "observation/wrist_image": dataset[100]["wrist_image"],
    "prompt": dataset[100]["prompt"],
}

example = data_input
noise = np.random.randn(1, 10, 32).astype(np.float32)

print(f"example: {example}")
# Run inference (same API as JAX)
action_chunk = policy.infer(example, noise=noise)["actions"]
print(f"action_chunk: {action_chunk}")

import json


# 将example和action_chunk保存到一个字典中
def convert_to_serializable(obj):
    """将numpy数组和PyTorch张量转换为可JSON序列化的格式."""
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

# 保存到文件
with open("inference_output_libero_torch.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("example和action_chunk已保存到 inference_output_libero_torch.json 文件中。")
