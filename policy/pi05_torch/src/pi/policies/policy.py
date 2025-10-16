"""Policy utilities and wrappers."""

import time
from typing import Any, TypeAlias
from collections.abc import Sequence

import numpy as np
import torch
from typing_extensions import override

import pi.transforms as _transforms
import pi.policies.base_policy as _base_policy
from pi.models import model as _model

BasePolicy: TypeAlias = _base_policy.BasePolicy


# 添加tree操作的工具函数
def tree_map(func, tree):
    """PyTorch版本的tree map，递归地对嵌套结构应用函数."""
    if isinstance(tree, dict):
        return {key: tree_map(func, value) for key, value in tree.items()}
    elif isinstance(tree, (list, tuple)):
        result = [tree_map(func, item) for item in tree]
        return type(tree)(result)
    else:
        return func(tree)


class Policy(BasePolicy):
    def __init__(
        self,
        # model: _model.BaseModel,
        model,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        self._model = self._model.to(pytorch_device)
        self._model.eval()
        self._sample_actions = model.sample_actions

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = tree_map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Make a batch and convert to torch.Tensor.
        def to_tensor_with_batch(x):
            if isinstance(x, np.ndarray):
                tensor = torch.from_numpy(x).to(self._pytorch_device)
            elif isinstance(x, torch.Tensor):
                tensor = x.to(self._pytorch_device)
            else:
                tensor = torch.tensor(x, device=self._pytorch_device)

            # Add batch dimension if not present
            if tensor.ndim == 0:  # scalar
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(0)  # Add batch dimension at the beginning
            return tensor

        # Make a batch and convert to jax.Array.
        # inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        # self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        inputs = tree_map(to_tensor_with_batch, inputs)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(self._pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        # Convert outputs back to numpy and remove batch dimension
        def tensor_to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x[0, ...].detach().cpu().numpy()  # Remove batch dimension and convert to numpy
            else:
                return np.asarray(x)

        outputs = tree_map(tensor_to_numpy, outputs)
        # outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @override
    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
