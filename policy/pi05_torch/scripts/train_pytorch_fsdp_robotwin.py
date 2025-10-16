"""FSDP training entrypoint that reuses train_pytorch.py logic and
follows FSDP practices outlined in scripts/fsdp.md.

Usage examples:
  torchrun --standalone --nproc_per_node=4 scripts/train_pytorch_fsdp_train.py \
      --configs pi05_airbot pi05_libero --exp_name run_fsdp --shuffle --num-workers 4

Notes:
  - Accepts a list of TrainConfig names via --configs; all must share model dims
    and global batch size. Model/optimizer/CKPT logic mirrors train_pytorch.py,
    while wrapping the model with FSDP.
  - DataLoader uses MultiLeRobotLoader + DistributedSampler per-rank.
"""

from __future__ import annotations

import os
import gc
import sys
import time
import math
import shutil
import logging
import platform
import dataclasses
import argparse
import pathlib
from typing import List, Tuple
import glob

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
import safetensors.torch
import tqdm
import wandb
from pi.training.config import TrainConfig, pi_config, LeRobotAirbotDataConfig, AssetsConfig, _optimizer, LeRobotRoboTwinDataConfig, DataConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

# Ensure `src` in path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ALL_FP32 = False  # global flag to disable mixed precision for debugging

import pi.training.config as _config  # noqa: E402
import pi.models_pytorch.pi0_pytorch  # noqa: E402
from pi.data import MultiLeRobotLoader, LeRobotDatasetConfig  # noqa: E402
from pi.models import model as _model  # noqa: E402


# ------------------------- logging / wandb utilities -------------------------
def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def log_memory_usage(device, step, phase="unknown"):
    if not torch.cuda.is_available():
        return
    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
    mem_resv = torch.cuda.memory_reserved(device) / 1e9
    mem_free = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1e9
    stats = torch.cuda.memory_stats(device)
    peak_alloc = stats.get("allocated_bytes.all.peak", 0) / 1e9
    peak_resv = stats.get("reserved_bytes.all.peak", 0) / 1e9
    ddp_info = f" | dist: rank={dist.get_rank()}, world={dist.get_world_size()}" if dist.is_initialized() else ""
    logging.info(
        f"Step {step} ({phase}): GPU mem alloc={mem_alloc:.2f}GB, resv={mem_resv:.2f}GB, free={mem_free:.2f}GB, "
        f"peak_alloc={peak_alloc:.2f}GB, peak_resv={peak_resv:.2f}GB{ddp_info}"
    )


def get_latest_checkpoint_step(checkpoint_dir: pathlib.Path) -> int | None:
    steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(steps) if steps else None


# ------------------------- dist + dataloader helpers -------------------------
# def _init_dist() -> Tuple[int, int, int, torch.device]:
#     if not dist.is_initialized():
#         backend = "nccl" if torch.cuda.is_available() else "gloo"
#         dist.init_process_group(backend=backend, init_method="env://")
#     rank = dist.get_rank() if dist.is_initialized() else 0
#     world = dist.get_world_size() if dist.is_initialized() else 1
#     local_rank = int(os.environ.get("LOCAL_RANK", rank))
#     if torch.cuda.is_available():
#         torch.cuda.set_device(local_rank)
#         device = torch.device(f"cuda:{local_rank}")
#     else:
#         device = torch.device("cpu")
#     if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
#         os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
#     return rank, world, local_rank, device


def _init_dist() -> Tuple[int, int, int, torch.device]:
    """Initialize torch.distributed (env://)."""
    is_initialized = dist.is_initialized()
    if not is_initialized:
        backend = "nccl"
        dist.init_process_group(backend=backend, init_method="env://")
        print(f"Initialized torch.distributed with backend={backend}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not is_initialized:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        else:
            raise RuntimeError("This script requires CUDA.")
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def _validate_shared_fields(configs: List[_config.TrainConfig]) -> None:
    if not configs:
        raise ValueError("configs must be non-empty")
    fields = (
        ("action_horizon", lambda c: c.model.action_horizon),
        ("action_dim", lambda c: c.model.action_dim),
        ("max_token_len", lambda c: c.model.max_token_len),
        ("discrete_state_input", lambda c: getattr(c.model, "discrete_state_input", False)),
        ("batch_size", lambda c: c.batch_size),
        ("pytorch_training_precision", lambda c: c.pytorch_training_precision),
    )
    for name, fn in fields:
        v0 = fn(configs[0])
        for c in configs[1:]:
            if fn(c) != v0:
                raise ValueError(f"All configs must share {name}. Got {v0} vs {fn(c)}")


def _build_dataset_configs(configs: List[_config.TrainConfig]) -> List[LeRobotDatasetConfig]:
    ds_cfgs: List[LeRobotDatasetConfig] = []
    for cfg in configs:
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        if dc.repo_id is None:
            raise ValueError(f"Repo ID not set for config '{cfg.name}'.")
        if dc.norm_stats is None:
            raise ValueError(f"Normalization stats missing for '{cfg.name}'. Run scripts/compute_norm_stats.py first.")
        ds_cfgs.append(
            LeRobotDatasetConfig(
                repo_id=dc.repo_id,
                action_sequence_keys=list(dc.action_sequence_keys),
                norm_stats=dc.norm_stats,
                use_quantile_norm=dc.use_quantile_norm,
            )
        )
    return ds_cfgs


def build_configs_from_parent_dir(
    parent_dir: str | pathlib.Path, template: _config.TrainConfig
) -> List[_config.TrainConfig]:
    """Build a list of TrainConfig by globbing first-level subdirectories.

    Only repo_id differs across configs; all other fields follow the template.
    """
    # NOTE(critical path): use glob for first-level dirs and keep absolute paths for repo_id
    pattern = os.path.join(str(parent_dir), "*/")
    candidates = sorted(glob.glob(pattern))
    subdirs = [p for p in candidates if os.path.isdir(p)]
    if not subdirs:
        raise FileNotFoundError(f"No first-level subdirectories found under: {parent_dir}")

    cfgs: List[_config.TrainConfig] = []
    for d in subdirs:
        abs_d = os.path.abspath(d)
        # Replace only the nested DataConfigFactory.repo_id while keeping other fields intact
        new_data = dataclasses.replace(template.data, repo_id=abs_d)
        new_cfg = dataclasses.replace(template, data=new_data)
        cfgs.append(new_cfg)
        logging.info(f"Built config for repo_id: {abs_d}")
    return cfgs


def create_distributed_dataloader(
    configs: List[_config.TrainConfig],
    *,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    # Create concatenated dataset and shard with DistributedSampler
    rank, world, _local_rank, _device = _init_dist()
    _validate_shared_fields(configs)
    base_model = configs[0].model
    global_batch = int(configs[0].batch_size)
    if global_batch % world != 0:
        raise ValueError(f"batch_size {global_batch} must be divisible by world_size {world}")
    local_batch = global_batch // world
    ds_cfgs = _build_dataset_configs(configs)

    multi_ds = MultiLeRobotLoader(
        datasets=ds_cfgs,
        batch_size=local_batch,
        action_horizon=int(base_model.action_horizon),
        action_dim=int(base_model.action_dim),
        max_token_len=int(base_model.max_token_len),
        discrete_state_input=bool(getattr(base_model, "discrete_state_input", False)),
    )
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = torch.utils.data.distributed.DistributedSampler(
        multi_ds, num_replicas=world, rank=rank, shuffle=shuffle, drop_last=True
    )
    return torch.utils.data.DataLoader(
        multi_ds,
        batch_size=local_batch,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=16,
        prefetch_factor=32,
        persistent_workers=(num_workers > 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=g,
    )


# ------------------------------ FSDP2 helpers ------------------------------


def _select_mp_policy_bf16() -> MixedPrecisionPolicy | None:
    # Prefer bf16 on Ampere+; otherwise fp16. Inputs will be cast at forward.
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if not ALL_FP32 else torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )


def _fsdp_wrap(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    mp_policy = _select_mp_policy_bf16()
    # paligemma = model.paligemma_with_expert.paligemma

    # Ensure uniform original parameter dtype (FSDP2 requires this at init).
    # Keep params in fp32 for stability; compute dtype is controlled by mp_policy.
    with torch.no_grad():
        model.to(torch.float32)
        # model.action_in_proj.to(torch.float32)
        # model.action_out_proj.to(torch.float32)
        # model.time_mlp_in.to(torch.float32)
        # model.time_mlp_out.to(torch.float32)
    # Exclude small projection heads from sharding (kept replicated).
    # ignored: set[torch.nn.Parameter] = set()
    # for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
    #     if hasattr(model, name):
    #         mod = getattr(model, name)
    #         if isinstance(mod, torch.nn.Module):
    #             ignored.update(p for p in mod.parameters(recurse=True))
    # logging.info(f"FSDP: Ignoring {len(ignored)} params from sharding (e.g. small heads)")
    # Broadcast full weights before sharding to keep identical init across ranks.
    if dist.is_initialized():
        with torch.no_grad():  # critical path: avoid autograd tracking
            for t in model.state_dict().values():
                if torch.is_tensor(t) and t.numel() > 0:
                    dist.broadcast(t, src=0)
    # fully_shard mutates module in-place; create optimizer AFTER this.
    fully_shard(module=model, mp_policy=mp_policy, reshard_after_forward=False)
    return model


def fsdp_save_full_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: _config.TrainConfig,
    data_config_first: _config.DataConfig | None,
    is_main: bool,
) -> None:
    # if not is_main:
    #     return
    if global_step <= 1:
        pass
    elif (global_step % config.save_interval != 0) and global_step != config.num_train_steps - 1:
        return
    else:
        pass

    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    # tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"
    # if tmp_ckpt_dir.exists():
    #     shutil.rmtree(tmp_ckpt_dir)
    # tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Use DCP state-dict helpers with FSDP2; keep single-file FULL semantics on rank0.
    # model_sd = get_model_state_dict(model)
    # optim_sd = get_optimizer_state_dict(model, optimizer)
    # state = {
    #     "model": model_sd,
    #     # "optim": optim_sd,
    #     "meta": {"global_step": global_step, "timestamp": time.time(), "config": dataclasses.asdict(config)},
    # }
    torch.distributed.checkpoint.save(model.state_dict(), checkpoint_id=final_ckpt_dir)
    # torch.distributed.checkpoint.load(state, checkpoint_id=tmp_ckpt_dir)

    # Optionally save norm stats of the first dataset for convenience
    try:
        from pi.shared import normalize as _normalize  # lazy import

        if data_config_first is not None and data_config_first.norm_stats is not None and data_config_first.asset_id:
            _normalize.save(final_ckpt_dir / "assets" / data_config_first.asset_id, data_config_first.norm_stats)
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Failed to save norm stats: {e!s}")

    # if final_ckpt_dir.exists():
    #     shutil.rmtree(final_ckpt_dir)
    # tmp_ckpt_dir.rename(final_ckpt_dir)
    logging.info(f"Saved FSDP FULL checkpoint at step {global_step} -> {final_ckpt_dir}")
    # if config.wandb_enabled:
    #     wandb.log({"checkpoint_step": global_step}, step=global_step)


def fsdp_load_full_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: pathlib.Path, device
) -> int:
    steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(steps)
    ckpt_dir = checkpoint_dir / f"{latest}"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest, "before_ckpt_load")

    ckpt = torch.load(ckpt_dir / "fsdp_full.pt", map_location="cpu")
    # Load on every rank; DCP maps full state to local shards.
    set_model_state_dict(model, ckpt["model"])  # no broadcast kw in API
    set_optimizer_state_dict(model, optimizer, ckpt["optim"])  # map to local shards
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest, "after_ckpt_load")
    logging.info(f"Loaded FSDP FULL checkpoint from step {latest}")
    return int(ckpt.get("meta", {}).get("global_step", latest))


# ------------------------------- Train entry -------------------------------
def _tree_map_to_device(item, target_device):
    # Recursively move nested structures to device
    if isinstance(item, dict):
        return {k: _tree_map_to_device(v, target_device) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        out = [_tree_map_to_device(v, target_device) for v in item]
        return type(item)(out)
    if hasattr(item, "__dict__") and not hasattr(item, "to"):
        import dataclasses as _dc

        if _dc.is_dataclass(item):
            new_attrs = {f.name: _tree_map_to_device(getattr(item, f.name), target_device) for f in _dc.fields(item)}
            return _dc.replace(item, **new_attrs)
        new_attrs = {n: _tree_map_to_device(v, target_device) for n, v in item.__dict__.items()}
        new_item = type(item).__new__(type(item))
        for n, v in new_attrs.items():
            setattr(new_item, n, v)
        return new_item
    return item.to(target_device) if hasattr(item, "to") else item


@torch.no_grad()
def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh=None,
) -> torch.Tensor:
    """Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)

    # Group gradients and parameters by device mesh to handle mixed meshes (e.g., EP + non-EP layers)
    mesh_to_grads = {}
    mesh_to_params = {}
    for p in parameters:
        if p.grad is not None:
            if isinstance(p.grad, DTensor):
                mesh_key = str(p.grad.device_mesh)
            else:
                # Regular tensors
                mesh_key = "local"

            if mesh_key not in mesh_to_grads:
                mesh_to_grads[mesh_key] = []
                mesh_to_params[mesh_key] = []
            mesh_to_grads[mesh_key].append(p.grad)
            mesh_to_params[mesh_key].append(p)

    # Compute total norm for each mesh group separately, then combine
    group_norms = []
    for grad_group in mesh_to_grads.values():
        group_norm = torch.nn.utils.get_total_norm(grad_group, norm_type, error_if_nonfinite, foreach)
        if isinstance(group_norm, DTensor):
            group_norm = group_norm.full_tensor()
        group_norms.append(group_norm)

    # Combine norms from different meshes
    if math.isinf(norm_type):
        total_norm = torch.stack(group_norms).max()
    else:
        total_norm_p = sum(norm**norm_type for norm in group_norms)
        total_norm = total_norm_p ** (1.0 / norm_type)

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    # Apply gradient clipping to each mesh group separately using the global total_norm
    for params_in_group in mesh_to_params.values():
        torch.nn.utils.clip_grads_with_norm_(params_in_group, max_norm, total_norm, foreach)

    return total_norm


def train_loop(
    configs: List[_config.TrainConfig],
    *,
    exp_name: str,
    shuffle: bool,
    num_workers: int,
    resume: bool,
    overwrite: bool,
    wandb_enabled: bool,
    frozen: bool = False,
):
    rank, world, local_rank, device = _init_dist()
    base = configs[0]
    _validate_shared_fields(configs)

    # Build base config with overridden exp_name; reuse all other settings
    config = dataclasses.replace(
        base, exp_name=exp_name, resume=resume, overwrite=overwrite, wandb_enabled=wandb_enabled
    )
    torch.manual_seed(config.seed + local_rank)
    np.random.seed(config.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + local_rank)

    is_main = (not dist.is_initialized()) or rank == 0

    # Prepare checkpoint dir
    resuming = False
    if config.resume:
        if config.checkpoint_dir.exists():
            latest = get_latest_checkpoint_step(config.checkpoint_dir)
            if latest is None:
                raise FileNotFoundError(f"No valid checkpoints in {config.checkpoint_dir} for resume")
            resuming = True
        else:
            raise FileNotFoundError(f"Checkpoint dir {config.checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
    if not resuming and is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Data: MultiLeRobotLoader + DistributedSampler
    data_loader = create_distributed_dataloader(configs, num_workers=num_workers, shuffle=shuffle, seed=config.seed)
    # Also keep a first dataset's DataConfig for assets saving
    first_data_config = configs[0].data.create(configs[0].assets_dirs, configs[0].model)

    # Model config dtype -> pytorch training precision
    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    with torch.device(device):
        raw_model = pi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)
        if frozen:
            for param in raw_model.paligemma_with_expert.paligemma.parameters():
                param.requires_grad = False
            logging.info(f"Created model {raw_model.__class__.__name__} on device {device}, with frozen weights")
    # Enable gradient checkpointing if available (before wrapping)
    if hasattr(raw_model, "gradient_checkpointing_enable"):
        raw_model.gradient_checkpointing_disable()

    # Optionally load non-FSDP weights on rank0, then broadcast on wrap
    if config.pytorch_weight_path and (not resuming):
        if is_main:
            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            missing, unexpected = safetensors.torch.load_model(raw_model, model_path, strict=False)
            if missing:
                logging.warning(f"Missing keys when loading model: {missing}")
            if unexpected:
                logging.warning(f"Unexpected keys when loading model: {unexpected}")
            logging.info(f"Loaded non-FSDP weights from: {config.pytorch_weight_path}")

    # Wrap with FSDP (must occur before optimizer creation)
    raw_model.paligemma_with_expert.embed_image = torch.compile(
        raw_model.paligemma_with_expert.embed_image, options={"triton.cudagraphs": False}
    )
    raw_model.paligemma_with_expert.embed_language_tokens = torch.compile(
        raw_model.paligemma_with_expert.embed_language_tokens, options={"triton.cudagraphs": False}
    )
    raw_model.paligemma_with_expert.forward = torch.compile(
        raw_model.paligemma_with_expert.forward, options={"triton.cudagraphs": False}
    )

    model = _fsdp_wrap(raw_model, device)

    # Optimizer and LR schedule
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim_params = (p for p in model.parameters() if p.requires_grad)
    optim = torch.optim.AdamW(
        optim_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Resume from FSDP FULL checkpoint if needed
    global_step = 0
    if resuming:
        global_step = fsdp_load_full_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed from step {global_step}")

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # Memory optimizations for large-scale runs
    if world >= 8 and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_wrap")

    # Training loop
    model.train()
    # model = torch.compile(model, options={"triton.cudagraphs": False})
    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="FSDP-Train", disable=not is_main)
    infos: list[dict] = []
    if is_main:
        logging.info(
            f"Host={platform.node()} world={world} local_rank={local_rank} batch={config.batch_size} steps={config.num_train_steps}"
        )

    while global_step < config.num_train_steps:
        if isinstance(data_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            data_loader.sampler.set_epoch(global_step // max(1, len(data_loader)))

        for batch in data_loader:
            if global_step >= config.num_train_steps:
                break
            # Our MultiLeRobotLoader yields a tree dict via collate_fn; move to device
            batch = _tree_map_to_device(batch, device)
            observation_dict = {k: v for k, v in batch.items() if k != "actions"}
            # Convert into Observation dataclass expected by the model
            observation = _model.Observation.from_dict(observation_dict)
            actions = batch["actions"].to(torch.float32)

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward/backward/step
            losses = model(observation, actions)
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(list(losses))
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, dtype=torch.float32, device=device)

            loss = losses.mean()
            loss.backward()
            total_grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
            optim.step()
            #
            #         if global_step >= 500:
            #             detached_loss = float(loss.detach())
            #             loss_is_good = torch.tensor(int(detached_loss <= 0.2), dtype=torch.int, device="cuda")
            #             dist.all_reduce(loss_is_good)
            #             all_loss_is_good = loss_is_good.item() == dist.get_world_size()
            #
            #             # Gradient clipping (global norm)
            #
            #             if not all_loss_is_good or total_grad_norm >= 0.8:
            #                 import pickle
            #                 from pathlib import Path
            #
            #                 logging.warning(
            #                     f"Abnormal step {global_step}: loss={loss.item():.4f}, grad_norm={total_grad_norm:.4f}"
            #                 )
            #                 abnormal_dir = Path(f"/data/robot/checkpoints/fp32_abn/abnormal_step_{global_step}_ckpt")
            #                 abnormal_dir.mkdir(parents=True, exist_ok=True)
            #                 torch.distributed.checkpoint.save(model.state_dict(), checkpoint_id=abnormal_dir)
            #                 pickle.dump(batch, open(abnormal_dir / "batch.pkl", "wb"))
            #                 logging.warning(f"Saved abnormal batch and model checkpoint for debugging.")
            #             else:
            #                 optim.step()
            #         else:
            optim.zero_grad(set_to_none=True)

            # Collect stats
            if is_main:
                infos.append(
                    {"loss": loss.item(), "lr": optim.param_groups[0]["lr"], "grad_norm": total_grad_norm.item()}
                )

            if is_main and (global_step % config.log_interval == 0):
                # elapsed = time.time() - (pbar.start_t or time.time())
                avg_loss = sum(i["loss"] for i in infos) / max(1, len(infos))
                avg_lr = sum(i["lr"] for i in infos) / max(1, len(infos))
                avg_grad_norm = sum(i["grad_norm"] for i in infos) / max(1, len(infos))
                # logging.info(f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s")
                if config.wandb_enabled:
                    wandb.log(
                        {"loss": avg_loss, "learning_rate": avg_lr, "total_grad_norm": avg_grad_norm},
                        step=global_step,
                    )
                infos.clear()

            global_step += 1
            fsdp_save_full_checkpoint(model, optim, global_step, config, first_data_config, is_main)
            if is_main:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}"})

    if is_main and config.wandb_enabled:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSDP training with MultiLeRobotLoader")
    # p.add_argument("--configs", nargs="+", type=str, default=["pi05_airbot"], help="List of TrainConfig names")
    p.add_argument("--exp_name", type=str, required=True, help="Experiment name (checkpoint subdir)")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument(
        "--frozen", dest="frozen", action="store_true", help="Use a frozen model (no training)", default=False
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Parent directory; glob first-level subdirectories as repo_id for configs",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=30_000)
    p.add_argument("--all-fp32", dest="all_fp32", action="store_true", help="Disable mixed precision", default=False)
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    init_logging()
    args = _parse_args(argv)
    # cfgs = [dataclasses.replace(_config.get_config(name)) for name in args.configs]
    # Base template; repo_id will be overridden when --data-root is provided.
    config0 = TrainConfig(
        name="pi05_robotwin",
        model=pi_config.PiConfig(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotRoboTwinDataConfig(
            repo_id="/home/jovyan/repo/Pi/data/adjust_bottle-50ep-agilex-demo_clean",
            # assets=AssetsConfig(asset_id="robotwin_agilex"),
            base_config=DataConfig(
                prompt_from_task=True,  
            ),
            extra_delta_transform=True,
        ),
        
        batch_size=int(args.batch_size),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,
            peak_lr=5e-5,
            decay_steps=int(args.train_steps) + 10_000,
            decay_lr=1e-5,
        ), 
        log_interval=10,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        save_interval=1000,
        checkpoint_base_dir="/data/robot/checkpoints/pi05",
        pytorch_weight_path="/data/private/robot/pi05_base_pytorch",
        num_train_steps=int(args.train_steps),
    )

    if args.data_root:
        cfgs = build_configs_from_parent_dir(args.data_root, config0)
    else:
        # Fallback to manual list if --data-root not provided
        cfgs = [
            dataclasses.replace(
                config0, data=dataclasses.replace(config0.data, repo_id="/home/jovyan/repo/Pi/data/adjust_bottle-50ep-agilex-demo_clean")
            ),
        ]

    if args.all_fp32:
        global ALL_FP32
        ALL_FP32 = True
        torch.backends.cuda.matmul.allow_tf32 = True  # 未来版本将逐步弃用
        torch.backends.cudnn.allow_tf32 = True

    train_loop(
        cfgs,
        exp_name=args.exp_name,
        shuffle=bool(args.shuffle),
        num_workers=int(args.num_workers),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        wandb_enabled=bool(args.wandb),
        frozen=bool(args.frozen),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
