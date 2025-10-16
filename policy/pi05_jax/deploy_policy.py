from .pi05_model import *


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
    if model.observation_window is None: # # observation_window是一个缓冲区/数据容器,用于存储当前时刻的观察数据,并传递给策略模型进行推理。
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)
    
    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    # ======== Get Action ========

    actions = model.get_action()[:model.pi05_step]
    
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
    

def reset_model(model):
    model.reset_observationwindows()
