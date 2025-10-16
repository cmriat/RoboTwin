"""Distributed DataLoader setup using MultiLeRobotLoader (PyTorch/FSDP-friendly).

Usage (example):
  torchrun --nproc_per_node=4 scripts/train_pytorch_fsdp.py \
      --configs pi05_airbot pi05_libero --num-workers 4 --shuffle

Notes:
  - Expects each TrainConfig to share model-related dimensions (action_horizon,
    action_dim, max_token_len, discrete_state_input) and global batch_size.
  - Builds a MultiLeRobotLoader over all datasets, then shards via DistributedSampler.
  - Only critical-path code contains English comments per repo guideline.
"""

from __future__ import annotations

import os
import sys
import argparse
import pathlib
from typing import List, Tuple

import torch
import torch.distributed as dist


# Ensure `src` is importable when executed as a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pi.data import MultiLeRobotLoader, LeRobotDatasetConfig  # noqa: E402
from pi.training import config as _config  # noqa: E402
from pi.training.data_loader import _collate_fn  # noqa: E402


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
    """Ensure model and batch hyperparams are identical across configs.
    Critical to avoid shape mismatch and batching inconsistencies.
    """
    if not configs:
        raise ValueError("configs must be a non-empty list")
    fields = (
        ("action_horizon", lambda c: c.model.action_horizon),
        ("action_dim", lambda c: c.model.action_dim),
        ("max_token_len", lambda c: c.model.max_token_len),
        ("discrete_state_input", lambda c: getattr(c.model, "discrete_state_input", False)),
        ("batch_size", lambda c: c.batch_size),
    )
    for name, fn in fields:
        val0 = fn(configs[0])
        for c in configs[1:]:
            if fn(c) != val0:
                raise ValueError(f"All configs must share {name}. Got {val0} vs {fn(c)}")


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


def create_distributed_dataloader(
    configs: List[_config.TrainConfig],
    *,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Create a per-rank DataLoader on top of MultiLeRobotLoader.

    - Builds a single MultiLeRobotLoader from all datasets.
    - Applies DistributedSampler to shard samples across ranks.
    - Uses repo's `_collate_fn` for nested stacking into torch.Tensors.
    """
    rank, world_size, _local_rank, _device = _init_dist()
    _validate_shared_fields(configs)

    model_cfg = configs[0].model
    global_batch = int(configs[0].batch_size)
    if global_batch % world_size != 0:
        raise ValueError(f"batch_size {global_batch} must be divisible by world_size {world_size}")
    local_batch = global_batch // world_size

    ds_cfgs = _build_dataset_configs(configs)

    # As a dataset (random access). PyTorch DataLoader will call __getitem__/__len__.
    multi_ds = MultiLeRobotLoader(
        datasets=ds_cfgs,
        batch_size=local_batch,  # not used by __getitem__, kept for __iter__ parity
        action_horizon=int(model_cfg.action_horizon),
        action_dim=int(model_cfg.action_dim),
        max_token_len=int(model_cfg.max_token_len),
        discrete_state_input=bool(getattr(model_cfg, "discrete_state_input", False)),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = torch.utils.data.distributed.DistributedSampler(
        multi_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )

    # Minimal, FSDP-friendly DataLoader (no persistent_workers if workers=0).
    loader = torch.utils.data.DataLoader(
        multi_ds,
        batch_size=local_batch,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=generator,
    )
    return loader


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize distributed MultiLeRobotLoader DataLoader")
    p.add_argument("--configs", nargs="+", type=str, default=["pi05_airbot"], help="List of config names")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batches", type=int, default=2, help="Preview N batches and exit")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    cfgs = [_config.get_config(name) for name in args.configs]
    rank, world_size, local_rank, device = _init_dist()

    dl = create_distributed_dataloader(cfgs, num_workers=args.num_workers, shuffle=args.shuffle, seed=args.seed)

    # Quick smoke: iterate a few batches; only rank 0 prints.
    print(
        f"[rank {rank}/{world_size} device={device}] "
        f"Previewing {args.batches} batches from distributed DataLoader..."
    )

    it = iter(dl)
    for i in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        # Only minimal structural prints to avoid log spam.
        actions = batch.get("actions")
        actions_shape = tuple(actions.shape) if hasattr(actions, "shape") else None
        print(f"batch[{i}] keys={sorted(batch.keys())} actions.shape={actions_shape}")

    if rank == 0:
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
