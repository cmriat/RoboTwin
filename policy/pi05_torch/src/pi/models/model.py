"""Model configuration and loading utilities."""

import abc
import enum
import logging
import dataclasses
import os
from pathlib import Path

import numpy as np
import torch
import safetensors.torch
import torch.distributed.checkpoint as dcp
import pi.ema_model
import pi.models_pytorch.pi0_pytorch as pi0_pytorch

EMA = True
logger = logging.getLogger("pi")


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models.

    Specific models should inherit from this class, and implement the `create` method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    def load_pytorch(self, train_config, weight_path: str):
        """Load PyTorch model from either safetensors or distributed checkpoint.

        Args:
            train_config: Training configuration
            weight_path: Path to model.safetensors file or directory containing distributed checkpoint

        Returns:
            Loaded PyTorch model
        """
        logger.info(f"train_config: {train_config}")

        # Create model first
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        

        # Check if weight_path is a file (safetensors) or directory (distributed checkpoint)
        weight_path_obj = Path(weight_path)

        if weight_path_obj.is_file() and weight_path_obj.suffix == ".safetensors":
            # Load from safetensors file
            logger.info(f"Loading model from safetensors: {weight_path}")
            missing_keys, unexpected_keys = safetensors.torch.load_model(model, weight_path, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")
        elif weight_path_obj.is_dir() or (weight_path_obj.parent.is_dir() and any(weight_path_obj.parent.glob("*.distcp"))):
            # Load from distributed checkpoint
            checkpoint_dir = weight_path_obj if weight_path_obj.is_dir() else weight_path_obj.parent
            print(f"Loading model from distributed checkpoint: {checkpoint_dir}")
            logger.info(f"Loading model from distributed checkpoint: {checkpoint_dir}")

            # Use cuda if available for faster loading
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading on device: {device}")
            logger.info(f"Loading on device: {device}")

            with torch.device(device):
                # Move model to device first
                model = model.to(device)
                # Load distributed checkpoint
                if EMA == True:
                    print("Loading EMA model")
                    dcp.load(model.state_dict(), checkpoint_id=str(checkpoint_dir))
                    ema_model = pi.ema_model.EMAModel(model)
                    dcp.load(ema_model.shadow, checkpoint_id=str(checkpoint_dir/"ema"))
                    ema_model.apply_shadow(model)
                else:
                    dcp.load(model.state_dict(), checkpoint_id=str(checkpoint_dir))

            print("✓ Successfully loaded distributed checkpoint")
            logger.info("Successfully loaded distributed checkpoint")
        else:
            raise FileNotFoundError(
                f"Invalid weight path: {weight_path}. "
                f"Expected either a .safetensors file or a directory containing distributed checkpoint (.distcp files)."
            )

        return model


import dataclasses
from typing import Dict, Generic, TypeVar, Optional

ArrayT = TypeVar("ArrayT")  # numpy.ndarray | torch.Tensor


@dataclasses.dataclass(frozen=True)  # frozen=True 保持不可变性
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model."""

    # 替换JAXtyping注解为标准类型提示
    images: Dict[str, ArrayT]  # 原: at.Float[ArrayT, "*b h w c"]
    image_masks: Dict[str, ArrayT]  # 原: at.Bool[ArrayT, "*b"]
    state: ArrayT  # 原: at.Float[ArrayT, "*b s"]

    # 可选字段
    tokenized_prompt: Optional[ArrayT] = None  # 原: at.Int[ArrayT, "*b l"]
    tokenized_prompt_mask: Optional[ArrayT] = None  # 原: at.Bool[ArrayT, "*b l"]
    token_ar_mask: Optional[ArrayT] = None  # 原: at.Int[ArrayT, "*b l"]
    token_loss_mask: Optional[ArrayT] = None  # 原: at.Bool[ArrayT, "*b l"]

    @classmethod
    def from_dict(cls, data: Dict) -> "Observation[ArrayT]":
        # from_dict 逻辑保持不变，只是类型注解改变
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        # 图像类型转换逻辑保持完全相同
        for key in data["image"]:
            if hasattr(data["image"][key], "dtype"):
                if data["image"][key].dtype == np.uint8:
                    data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                    data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> Dict:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Actions = at.Float[ArrayT, "*b ah ad"]
Actions = ArrayT  # Shape: (*batch, action_horizon, action_dim)
