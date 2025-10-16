"""Inference script for Pi models."""

import pi.policies.droid_policy as droid_policy
from pi.policies import policy_config_torch
from pi.training import config as _config

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config_torch.create_trained_policy(config, checkpoint_dir)

example = droid_policy.make_droid_example()
# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
