from .pi05_model import *
import numpy as np
import os


# # Global variable to track if this is the first episode
# _first_episode = True
# _actions_buffer = []


def encode_obs(observation): 
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state



def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi05_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi05_step"])
    return PI05(train_config_name, model_name, checkpoint_id, pi05_step)


def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    # global _first_episode, _actions_buffer

    if model.observation_window is None: # # observation_window是一个缓冲区/数据容器,用于存储当前时刻的观察数据,并传递给策略模型进行推理。
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    # ======== Get Action ========

    actions = model.get_action()[:model.pi05_step]

    # # Save actions if this is the first episode
    # if _first_episode:
    #     for action in actions:
    #         _actions_buffer.append(action.copy())

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
    

def reset_model(model):
    # global _first_episode, _actions_buffer

    # # Save actions after the first episode
    # if _first_episode and len(_actions_buffer) > 0:
    #     save_dir = "./eval_actions"
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, "first_episode_actions.npy")

    #     # Convert list of actions to numpy array
    #     actions_array = np.array(_actions_buffer)

    #     # Save to file
    #     np.save(save_path, actions_array)
    #     print(f"\n{'='*60}")
    #     print(f"First episode actions saved to: {save_path}")
    #     print(f"Total actions saved: {len(_actions_buffer)}")
    #     print(f"Action shape: {actions_array.shape}")
    #     print(f"{'='*60}\n")

    #     # Mark that we've saved the first episode
    #     _first_episode = False
    #     _actions_buffer = []

    model.reset_observationwindows()
