"""PyTorch policy configuration."""

import os
import logging
import pathlib
from typing import Any

import pi.transforms as transforms
import pi.policies.policy as _policy
import pi.shared.download as download
from pi.training import config as _config, checkpoints as _checkpoints


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a Torch policy from a trained checkpoint.

    Args:
        train_config: Training config used to construct the model.
        checkpoint_dir: Directory containing the Torch checkpoint files.
        repack_transforms: Optional transforms applied before other transforms.
        sample_kwargs: Optional kwargs passed to `sample_actions`.
        default_prompt: Default prompt to inject if missing in inputs.
        norm_stats: Optional normalization stats; loaded from checkpoint if None.
        pytorch_device: Device string (e.g., "cpu", "cuda", "cuda:0"). If None, picks
            "cuda" when available, else "cpu".
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Try to load from safetensors first, fallback to distributed checkpoint
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(weight_path):
        logging.info("Loading PyTorch model from safetensors...")
        model = train_config.model.load_pytorch(train_config, weight_path)
    else:
        # Try to load as distributed checkpoint
        logging.info("Safetensors not found, attempting to load as distributed checkpoint...")
        model = train_config.model.load_pytorch(train_config, str(checkpoint_dir))
    # Use bfloat16 where appropriate while keeping numerically sensitive params in float32.
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # Load norm stats from checkpoint to match training.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Select device if not provided
    if pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device=pytorch_device,
    )
