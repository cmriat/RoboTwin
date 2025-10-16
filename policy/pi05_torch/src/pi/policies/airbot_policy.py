"""Airbot policy implementation."""

import dataclasses

import numpy as np
import einops

from pi import transforms
from pi.models import model as _model


def make_airbot_example() -> dict:
    """Creates a random input example for the Airbot policy."""
    return {
        "observation/state": np.random.rand(14),
        "observation/cam_env": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/cam_left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/cam_right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):
    """This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For Airbot dataset with 3 cameras and dual-arm setup.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["observation/cam_env"])
        left_wrist_image = _parse_image(data["observation/cam_left_wrist"])
        right_wrist_image = _parse_image(data["observation/cam_right_wrist"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):
    """This class is used to convert outputs from the model back to the dataset specific format.

    It is used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 actions for Airbot dual-arm setup.
        return {"actions": np.asarray(data["actions"][:, :14])}
