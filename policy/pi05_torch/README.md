# pi

A Pure PyTorch Implementation of pi

## Installation
1. pixi安装：引入lerobot需要GIT_LFS_SKIP_SMUDGE=1
    ```bash
    GIT_LFS_SKIP_SMUDGE=1 pixi install
    ```

## Checkpoints Path
### Base Models
- Pi0: gs://openpi-assets/checkpoints/pi0_base
- Pi05: gs://openpi-assets/checkpoints/pi05_base
- pi05_libero: gs://openpi-assets/checkpoints/pi05_libero

### JAX转Pytorch参考openpi官方文档：
[Converting JAX Models to PyTorch](https://github.com/Physical-Intelligence/openpi/blob/main/README.md#converting-jax-models-to-pytorch)


## Airbot双臂数据推理流程

### 1. 数据集准备
确保你的LeRobot数据集结构如下：
```
/data/robot/0922_250samples_merge/
├── meta/
│   ├── info.json
│   └── stats.json
├── data/
└── videos/
```

### 2. 转换normalization统计信息
```bash
python scripts/convert_lerobot_stats.py
```
这会将LeRobot格式的stats.json转换为Pi模型期望的norm_stats.json格式：
- `observation.state` → `state`
- `action` → `actions`

### 3. 运行推理
```bash
python scripts/inference_airbot.py
```
/home/jovyan/repo/Pi/src/pi/data.py这个也要改





# 修改configuration
src/pi/policies/robotwin_policy.py
src/pi/training/config.py