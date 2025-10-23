#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import numpy as np
from pathlib import Path
from pi.models import model as _model
from pi.policies import aloha_policy
from pi.policies import policy_config_torch as _policy_config
from pi.shared import download
from pi.training import config as _config
from pi.training import data_loader as _data_loader
import pi.shared.normalize as _normalize
import os
import cv2
from PIL import Image

from pi.models import model as _model
from pi.policies import policy_config_torch as _policy_config
from pi.shared import download
from pi.training import config as _config
from pi.training import data_loader as _data_loader


class PI05:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi05_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        
        if os.path.isabs(self.model_name):
            checkpoint_path = f"{self.model_name}/{self.checkpoint_id}"
        else:
            error_message = f"model_name must be an absolute path, but got {self.model_name}"
            raise ValueError(error_message)
        
        # Load normalization statistics
        norm_stats_path = "/home/jovyan/repo/openpi/data/beat_block_hammer-50ep-agilex-demo_clean"
        norm_stats = _normalize.load(norm_stats_path)
        
        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_path,
            norm_stats=norm_stats)
        print("loading model success!")
        
        self.observation_window = None
        self.pi05_step = pi05_step


    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "observation/state": state,
            "observation/head_image": img_front,
            "observation/left_wrist_image": img_left,
            "observation/right_wrist_image": img_right,
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_observationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
