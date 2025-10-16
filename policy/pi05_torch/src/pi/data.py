"""Bare-bones LeRobot Airbot data loader without transform abstractions.

Also includes a multi-dataset variant that can iterate and batch
across multiple LeRobot repos using a simple per-dataset config.
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses


import einops
import numpy as np
import torch

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from pi.models import tokenizer as tokenizer_mod
from pi.training import config as _config
from pi.shared import image_tools
import logging


def _repack_airbot(sample: dict) -> dict:
    # Support both cam_env (Airbot) and cam_high (RoboTwin) as the base camera key
    if "observation.images.cam_env" in sample:
        base_cam = sample["observation.images.cam_env"]
    elif "observation.images.cam_high" in sample:
        base_cam = sample["observation.images.cam_high"]
    else:
        raise KeyError("Neither 'observation.images.cam_env' nor 'observation.images.cam_high' found in sample")

    result: dict[str, object] = {
        "observation/cam_env": base_cam,
        "observation/cam_left_wrist": sample["observation.images.cam_left_wrist"],
        "observation/cam_right_wrist": sample["observation.images.cam_right_wrist"],
        "observation/state": sample["observation.state"],
    }
    if "action" in sample:
        result["actions"] = sample["action"]
    if "task" in sample:
        result["prompt"] = sample["task"]
    elif "prompt" in sample:
        result["prompt"] = sample["prompt"]
    return result


def _airbot_inputs(data: dict) -> dict:
    def _to_uint8_image(array: np.ndarray) -> np.ndarray:
        image = np.asarray(array)
        if np.issubdtype(image.dtype, np.floating):
            image = (255.0 * image).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        return image.astype(np.uint8, copy=False)

    result = {
        "state": np.asarray(data["observation/state"]),
        "image": {
            "base_0_rgb": _to_uint8_image(data["observation/cam_env"]),
            "left_wrist_0_rgb": _to_uint8_image(data["observation/cam_left_wrist"]),
            "right_wrist_0_rgb": _to_uint8_image(data["observation/cam_right_wrist"]),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "prompt": data.get("prompt"),
    }
    if "actions" in data:
        result["actions"] = np.asarray(data["actions"])
    return result


def _normalize_array(x: np.ndarray, stats, use_quantiles: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile stats required when use_quantiles=True")
        q01 = stats.q01[..., : x.shape[-1]]
        q99 = stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    mean = stats.mean[..., : x.shape[-1]]
    std = stats.std[..., : x.shape[-1]]
    return (x - mean) / (std + 1e-6)




def _resize_images(data: dict, height: int, width: int) -> dict:
    result = dict(data)
    result["image"] = {key: image_tools.resize_with_pad(img, height, width) for key, img in data["image"].items()}
    return result


def _tokenize_prompt(data: dict, tokenizer, *, discrete_state_input: bool) -> dict:
    result = dict(data)
    prompt = result.pop("prompt", None)
    if prompt is None:
        raise ValueError("Prompt is required for tokenization.")
    if not isinstance(prompt, str):
        prompt = str(prompt if np.isscalar(prompt) else prompt.item())
    state_arg = result["state"] if discrete_state_input else None
    tokens, mask = tokenizer.tokenize(prompt, state_arg)
    result["tokenized_prompt"] = tokens
    result["tokenized_prompt_mask"] = mask
    return result

def _normalize(data: dict, norm_stats: dict, use_quantiles: bool) -> dict:
    result = dict(data)
    if "state" in result and "state" in norm_stats:
        result["state"] = _normalize_array(result["state"], norm_stats["state"], use_quantiles)
    if "actions" in result and "actions" in norm_stats:
        result["actions"] = _normalize_array(result["actions"], norm_stats["actions"], use_quantiles)
    return result

def _pad_state_actions(data: dict, target_dim: int) -> dict:
    def _pad_last_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
        array = np.asarray(array, dtype=np.float64)
        if array.shape[-1] >= target_dim:
            return array
        pad_width = [(0, 0)] * array.ndim
        pad_width[-1] = (0, target_dim - array.shape[-1])
        return np.pad(array, pad_width, constant_values=0.0)

    result = dict(data)
    result["state"] = _pad_last_dim(result["state"], target_dim)
    if "actions" in result:
        result["actions"] = _pad_last_dim(result["actions"], target_dim)
    return result


def _stack_tree(items: list[dict]) -> dict:
    def stack(*values):
        first = values[0]
        if isinstance(first, dict):
            return {key: stack(*[value[key] for value in values]) for key in first}
        if isinstance(first, (list, tuple)):
            packed = [stack(*[value[idx] for value in values]) for idx in range(len(first))]
            return type(first)(packed)
        return torch.stack([torch.as_tensor(value) for value in values], dim=0)

    return stack(*items)


class SimpleLeRobotLoader:
    """Minimal iterator that batches LeRobot Airbot samples.

    Note: `data_config` is intentionally not used here. Only the fields that were
    previously read from `data_config` are accepted directly via constructor
    parameters to avoid requiring a full TrainConfig initialization.
    """

    def __init__(
        self,
        config: _config.TrainConfig | None = None,
        *,
        # Fields previously sourced from data_config
        repo_id: str | None = None,
        action_sequence_keys: list[str] | None = None,
        norm_stats: dict | None = None,
        use_quantile_norm: bool | None = None,
        # Fields previously sourced from model/batch config
        batch_size: int | None = None,
        action_horizon: int | None = None,
        action_dim: int | None = None,
        max_token_len: int | None = None,
        discrete_state_input: bool | None = None,
    ) -> None:
        # Support both explicit-args path and legacy config path.
        if config is not None:
            # Extract non-data fields from TrainConfig.
            batch_size = config.batch_size if batch_size is None else batch_size
            action_horizon = config.model.action_horizon if action_horizon is None else action_horizon
            action_dim = config.model.action_dim if action_dim is None else action_dim
            max_token_len = config.model.max_token_len if max_token_len is None else max_token_len
            if discrete_state_input is None:
                discrete_state_input = getattr(config.model, "discrete_state_input", False)

            # Temporarily build data_config to fetch needed fields, but do not store it.
            tmp_dc = config.data.create(config.assets_dirs, config.model)
            repo_id = tmp_dc.repo_id if repo_id is None else repo_id
            action_sequence_keys = (
                list(tmp_dc.action_sequence_keys) if action_sequence_keys is None else action_sequence_keys
            )
            norm_stats = tmp_dc.norm_stats if norm_stats is None else norm_stats
            use_quantile_norm = tmp_dc.use_quantile_norm if use_quantile_norm is None else use_quantile_norm

        # Validate required fields for explicit construction.
        if repo_id is None:
            raise ValueError("repo_id must be set")
        if norm_stats is None:
            raise ValueError("Normalization stats are required.")
        if action_sequence_keys is None:
            raise ValueError("action_sequence_keys must be set")
        if batch_size is None or action_horizon is None or action_dim is None or max_token_len is None:
            raise ValueError("batch_size, action_horizon, action_dim, and max_token_len must be provided")
        if use_quantile_norm is None:
            use_quantile_norm = False
        if discrete_state_input is None:
            discrete_state_input = False

        # Store compact state; avoid keeping train_config/data_config around.
        self.repo_id = repo_id
        self.action_sequence_keys = list(action_sequence_keys)
        self.norm_stats = norm_stats
        self.use_quantile_norm = use_quantile_norm
        self.batch_size = int(batch_size)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.max_token_len = int(max_token_len)
        self.discrete_state_input = bool(discrete_state_input)

        metadata = lerobot_dataset.LeRobotDatasetMetadata(self.repo_id)
        # Use dataset fps to build per-key delta timestamps of length action_horizon.
        delta_timestamps = {
            key: [t / metadata.fps for t in range(self.action_horizon)] for key in self.action_sequence_keys
        }
        self.dataset = lerobot_dataset.LeRobotDataset(self.repo_id, delta_timestamps=delta_timestamps)

        self.tokenizer = tokenizer_mod.PaligemmaTokenizer(self.max_token_len)

    def _transform(self, sample: dict) -> dict:
        step = _repack_airbot(sample)
        step = _airbot_inputs(step)
        step = _normalize(step, self.norm_stats, self.use_quantile_norm)
        step = _resize_images(step, 224, 224)
        step = _tokenize_prompt(step, self.tokenizer, discrete_state_input=self.discrete_state_input)
        step = _pad_state_actions(step, self.action_dim)
        return step

    # Random-access API to fetch a single transformed item
    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        return self._transform(sample)

    # Dataset length passthrough
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict]:
        batch_size = self.batch_size
        buffer: list[dict] = []
        for sample in self.dataset:
            buffer.append(self._transform(sample))
            if len(buffer) == batch_size:
                yield _stack_tree(buffer)
                buffer = []
        if buffer:
            yield _stack_tree(buffer)


# Per-dataset configuration for the multi-dataset loader.
@dataclasses.dataclass(frozen=True)
class LeRobotDatasetConfig:
    repo_id: str
    action_sequence_keys: list[str]
    norm_stats: dict
    use_quantile_norm: bool = False


class MultiLeRobotLoader:
    """Round-robin iterator that batches samples from multiple LeRobot repos.

    The global batching/model params are provided once, while dataset-specific
    parameters are carried by `datasets` (previously passed as non-config args).
    """

    def __init__(
        self,
        *,
        datasets: list[LeRobotDatasetConfig],
        batch_size: int,
        action_horizon: int,
        action_dim: int,
        max_token_len: int,
        discrete_state_input: bool = False,
    ) -> None:
        if not datasets:
            raise ValueError("datasets must be a non-empty list")

        self.batch_size = int(batch_size)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.max_token_len = int(max_token_len)
        self.discrete_state_input = bool(discrete_state_input)

        # Build multiple SimpleLeRobotLoader instances (source of truth).
        self._loaders: list[SimpleLeRobotLoader] = []
        for cfg in datasets:
            loader = SimpleLeRobotLoader(
                None,
                repo_id=cfg.repo_id,
                action_sequence_keys=list(cfg.action_sequence_keys),
                norm_stats=cfg.norm_stats,
                use_quantile_norm=cfg.use_quantile_norm,
                batch_size=self.batch_size,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                max_token_len=self.max_token_len,
                discrete_state_input=self.discrete_state_input,
            )
            self._loaders.append(loader)

        # Precompute index offsets for O(log N) __getitem__ lookup across loaders.
        self._offsets: list[int] = [0]
        total = 0
        for ld in self._loaders:
            total += len(ld)
            self._offsets.append(total)

        self.valid_ptr = None

    # Flattened length across all sub-loaders
    def __len__(self) -> int:  # type: ignore[override]
        return self._offsets[-1]

    # Random-access across concatenated datasets
    def __getitem__(self, idx: int) -> dict:
        i = None
        local_idx = None
        try:
            if idx < 0:
                idx = len(self) + idx
            if idx < 0 or idx >= len(self):
                raise IndexError("index out of range")
            # Binary search in prefix sums; returns rightmost insertion point
            import bisect

            i = bisect.bisect_right(self._offsets, idx) - 1
            local_idx = idx - self._offsets[i]
            data = self._loaders[i][local_idx]
            self.valid_ptr = (i, local_idx)  # remember last valid loader index
            return data
        except Exception as e:
            repo_id = self._loaders[i].repo_id if i is not None else "unknown"
            msg = f"Error fetching index {idx}, repo_id {repo_id}, local_idx {local_idx}: {type(e).__name__}: {e}"
            logging.error(msg)
            # Fallback to a previously valid sample to avoid crashing the training
            i, local_idx = self.valid_ptr if self.valid_ptr is not None else (0, 0)
            return self._loaders[i][local_idx]

    def __iter__(self) -> Iterator[dict]:
        # Round-robin over per-dataset raw iterators, using each loader's transform.
        iters = [(iter(ld.dataset), ld._transform) for ld in self._loaders]
        active_idx = list(range(len(iters)))
        buffer: list[dict] = []
        rr = 0
        while active_idx:
            i = active_idx[rr % len(active_idx)]
            it, tx = iters[i]
            try:
                sample = next(it)
            except StopIteration:
                active_idx.pop(rr % len(active_idx))
                continue
            buffer.append(tx(sample))
            rr += 1
            if len(buffer) == self.batch_size:
                yield _stack_tree(buffer)
                buffer = []
        if buffer:
            yield _stack_tree(buffer)
