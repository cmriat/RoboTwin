
import os, torch, torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, BackwardPrefetch, MixedPrecision
)

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# 你的模型
model = build_model()  # <- 原来怎么写就怎么来
model = model.to(device)

# 建议：bf16 可用优先，否则 fp16
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
mp = MixedPrecision(
    param_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    reduce_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    buffer_dtype=torch.float32,
)

# ⚠️ 先 wrap 再创建 optimizer（FSDP 要求）
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 等价
    device_id=device if device.type == "cuda" else None,
    use_orig_params=True,           # 便于从“非FSDP权重”加载
    sync_module_states=True,        # 首次wrap时从rank0广播权重
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    mixed_precision=mp,
    limit_all_gathers=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

一定要在 FSDP 包装之后再创建优化器，因为 FSDP 会替换参数张量（flat 参数）。
PyTorch Docs

sync_module_states=True 可在 wrap 当下把 rank0 的参数广播给其它 rank，尤其当你在 wrap 之前只在 rank0 加载了权重时非常有用。
PyTorch Docs

如果是 Transformer 等大模型，推荐用自动 wrap（见下一小节）；小模型一个大 FSDP 也能跑。

2) 怎么存 ckpt？

FSDP 存档有两条主路：full（单文件，rank0 保存）和sharded（分片，每个 rank 各存一份）。官方建议 full 存档时开启 offload_to_cpu & rank0_only 来省显存/内存；而大模型训练断点续训一般用 sharded。
PyTorch Docs

下面给你两个函数对（存/载），把模型、优化器、以及自定义额外信息一起管起来。

import torch, os
from torch.distributed.fsdp import (
    StateDictType,
    FullStateDictConfig, FullOptimStateDictConfig,
    FullyShardedDataParallel as FSDP,
)

def save_full_ckpt(path, model: FSDP, optimizer, extra: dict = None):
    """只在 rank0 落盘一个文件"""
    is_main = (not dist.is_initialized()) or dist.get_rank() == 0

    sd_cfg  = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    osd_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # 在上下文中，model.state_dict() 会被“聚合到CPU”，且只有 rank0 拿到内容
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_cfg, osd_cfg):
        state = {
            "model": model.state_dict(),
            "optim": FSDP.optim_state_dict(model, optimizer),
            "extra": extra or {},
        }
    if is_main:
        torch.save(state, path)

def load_full_ckpt(path, model: FSDP, optimizer):
    sd_cfg  = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    osd_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    # 设定 FULL 策略以匹配我们保存时的格式
    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_cfg, osd_cfg)

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    # 把“FULL/UNSHARDED 优化器状态”转换成 FSDP 优化器可加载的扁平格式
    flat_osd = FSDP.optim_state_dict_to_load(model, optimizer, ckpt["optim"])
    optimizer.load_state_dict(flat_osd)
    return ckpt.get("extra", {})

3) 从“非 FSDP 权重”加载到 FSDP

两种常见做法：

A. 先在 rank0 加载到“未wrap的原模型”，再用 sync_module_states=True wrap：

raw_model = build_model()
if dist.get_rank() == 0:
    raw_model.load_state_dict(torch.load("vanilla_model.pt", map_location="cpu"))  # 单进程加载

fsdp_model = FSDP(
    raw_model.to(device),
    device_id=device,
    use_orig_params=True,
    sync_module_states=True,   # wrap 时自动把 rank0 的权重广播给其它 rank
)
这正是官方文档推荐的套路。
PyTorch Docs

B. 或者：wrap 之后用 FULL state_dict 直接 load_state_dict（见 2.1 的 load_full_ckpt）。

4) 常见易错点

先 wrap 再建 optimizer（很关键）。
PyTorch Docs

device_id 指定到本 rank 的 GPU；有 sync_module_states=True 时需要确保模块在 GPU 或者传入 device_id。
PyTorch Docs

存/载一定放到 FSDP.state_dict_type(...) 上下文（或 set_state_dict_type）里，否则拿到的不是你想要的格式。
PyTorch Docs
+1

优化器状态要先过一遍 FSDP.optim_state_dict_to_load(...) 再 optimizer.load_state_dict(...)。

