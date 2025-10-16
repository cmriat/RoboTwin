"""Minimal dataset debugger that mirrors the training DataLoader setup.

Run with torchrun to match distributed DataLoader behavior. On any data
decoding error inside worker processes, it prints which underlying dataset
(`repo_id`) and index caused the failure, then re-raises to stop early.

Examples:
  pixi run torchrun --nnodes 1 --nproc_per_node 8 \
    scripts/debug_dataloader_sources.py --exp_name debug --data-root /data/robot/merged \
    --max-batches 2 --shuffle

Notes:
  - Comments are in English and placed only on critical paths.
  - No model or training is created; this only builds the DataLoader.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import logging
import os
import pathlib
import sys
from typing import List, Tuple

import torch
import torch.distributed as dist

# Ensure `src` in path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pi.training.config as _config  # noqa: E402
from pi.data import MultiLeRobotLoader, LeRobotDatasetConfig  # noqa: E402


# ------------------------------ logging helpers ------------------------------
def init_logging() -> None:
    class _F(logging.Formatter):
        def format(self, record):
            record.levelname = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E"}.get(
                record.levelname, record.levelname
            )
            return super().format(record)

    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S")
    logging.getLogger().handlers[0].setFormatter(_F(fmt, datefmt="%H:%M:%S"))


# ------------------------- dist + dataloader helpers -------------------------
def _init_dist() -> Tuple[int, int, int, torch.device]:
    """Initialize torch.distributed (env://) and select local CUDA device."""
    if not dist.is_initialized():
        backend = "nccl"
        dist.init_process_group(backend=backend, init_method="env://")
        print(f"Initialized torch.distributed with backend={backend}")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA for parity with training setup.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world, local_rank, device


def _validate_shared_fields(configs: List[_config.TrainConfig]) -> None:
    fields = (
        ("action_horizon", lambda c: c.model.action_horizon),
        ("action_dim", lambda c: c.model.action_dim),
        ("max_token_len", lambda c: c.model.max_token_len),
        ("discrete_state_input", lambda c: getattr(c.model, "discrete_state_input", False)),
        ("batch_size", lambda c: c.batch_size),
        ("pytorch_training_precision", lambda c: c.pytorch_training_precision),
    )
    v0 = [fn(configs[0]) for _, fn in fields]
    for cfg in configs[1:]:
        v = [fn(cfg) for _, fn in fields]
        if v != v0:
            raise ValueError("All TrainConfig must share core fields for distributed loading.")


def _build_dataset_configs(configs: List[_config.TrainConfig]) -> List[LeRobotDatasetConfig]:
    ds_cfgs: List[LeRobotDatasetConfig] = []
    for cfg in configs:
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        if dc.repo_id is None:
            raise ValueError(f"Repo ID not set for config '{cfg.name}'.")
        if dc.norm_stats is None:
            raise ValueError(f"Normalization stats missing for '{cfg.name}'.")
        ds_cfgs.append(
            LeRobotDatasetConfig(
                repo_id=dc.repo_id,
                action_sequence_keys=list(dc.action_sequence_keys),
                norm_stats=dc.norm_stats,
                use_quantile_norm=dc.use_quantile_norm,
            )
        )
    return ds_cfgs


class DebugMultiLeRobotLoader(MultiLeRobotLoader):
    """Wraps MultiLeRobotLoader to annotate which dataset triggers failures.

    Critical path: Catch exceptions at map-style `__getitem__` so DataLoader
    worker errors include repo_id/local/global indices.
    """

    def __getitem__(self, idx: int):  # type: ignore[override]
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        import bisect

        i = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[i]
        try:
            return self._loaders[i][local_idx]
        except Exception as e:  # noqa: BLE001
            # Only essential context to identify the bad data source and position
            repo = getattr(self._loaders[i], "repo_id", f"<unknown_loader_{i}>")
            msg = (
                f"Data decode error from repo '{repo}' | loader={i} local_idx={local_idx} global_idx={idx}."
            )
            logging.error(msg)


def create_distributed_dataloader(
    configs: List[_config.TrainConfig], *, shuffle: bool = False, seed: int = 0, persistent_workers: bool = False
) -> torch.utils.data.DataLoader:
    # Mirror scripts/train_pytorch_fsdp.py exactly.
    rank, world, _local_rank, _device = _init_dist()
    _validate_shared_fields(configs)
    base_model = configs[0].model
    global_batch = int(configs[0].batch_size)
    if global_batch % world != 0:
        raise ValueError(f"batch_size {global_batch} must be divisible by world_size {world}")
    local_batch = global_batch // world
    ds_cfgs = _build_dataset_configs(configs)

    multi_ds = DebugMultiLeRobotLoader(
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
        # Mirror training script: gate persistence off the flag
        persistent_workers=bool(persistent_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=g,
    )


def build_configs_from_parent_dir(parent_dir: str | pathlib.Path, template: _config.TrainConfig) -> List[_config.TrainConfig]:
    pattern = os.path.join(str(parent_dir), "*/")
    candidates = sorted(glob.glob(pattern))
    subdirs = [p for p in candidates if os.path.isdir(p)]
    if not subdirs:
        raise FileNotFoundError(f"No first-level subdirectories found under: {parent_dir}")
    cfgs: List[_config.TrainConfig] = []
    for d in subdirs:
        abs_d = os.path.abspath(d)
        new_data = dataclasses.replace(template.data, repo_id=abs_d)
        cfgs.append(dataclasses.replace(template, data=new_data))
        logging.info(f"Built config for repo_id: {abs_d}")
    return cfgs


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset debugger with MultiLeRobotLoader")
    p.add_argument("--exp_name", type=str, required=True, help="Experiment name (unused; parity with train script)")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--data-root", type=str, default=None, help="Glob first-level subdirs as repo_id")
    p.add_argument("--repos", nargs="*", default=None, help="Explicit repo_id paths (overrides --data-root)")
    p.add_argument("--max-batches", type=int, default=2, help="Number of batches to fetch before exiting")
    p.add_argument("--persistent-workers", action="store_true", help="Enable persistent_workers like training")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    init_logging()
    args = _parse_args(sys.argv[1:])

    # Base template (matching train script defaults)
    config0 = _config.TrainConfig(
        name="pi05_airbot",
        model=_config.pi_config.PiConfig(pi05=True, action_horizon=10, discrete_state_input=False),
        data=_config.LeRobotAirbotDataConfig(
            repo_id="/data/robot/PLACEHOLDER",
            assets=_config.AssetsConfig(asset_id="airbot"),
        ),
        save_interval=1000,
        checkpoint_base_dir="/data/robot/checkpoints/pi05",
        batch_size=64,
        lr_schedule=_config._optimizer.CosineDecaySchedule(
            warmup_steps=3000,
            peak_lr=3e-5,
            decay_steps=40_000,
            decay_lr=1e-5,
        ),
        log_interval=10,
        optimizer=_config._optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=None,
        num_train_steps=10,
    )

    if args.repos:
        cfgs = [dataclasses.replace(config0, data=dataclasses.replace(config0.data, repo_id=path)) for path in args.repos]
    elif args.data_root:
        cfgs = build_configs_from_parent_dir(args.data_root, config0)
    else:
        # Fallback: keep parity with train script when no data-root is specified
        cfgs = [
            dataclasses.replace(config0, data=dataclasses.replace(config0.data, repo_id="/data/robot/0922_250samples_merge")),
            dataclasses.replace(config0, data=dataclasses.replace(config0.data, repo_id="/data/robot/0925_401samples_merge")),
        ]

    # Build DataLoader (distributed) and attempt to pull a few batches
    loader = create_distributed_dataloader(
        cfgs, shuffle=bool(args.shuffle), seed=0, persistent_workers=bool(args.persistent_workers)
    )
    # Critical path: iterate to trigger worker-side decoding; catch high-level errors too
    from tqdm.auto import tqdm
    try:
        fetched = 0
        for _ in tqdm(loader):
            fetched += 1
            # if fetched >= int(args.max_batches):
            #     break
    except Exception:
        # Let the enriched RuntimeError from DebugMultiLeRobotLoader bubble up
        raise
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
