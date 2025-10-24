import json
import os
import pathlib
import random
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import torch


def create_empty_lerobot(input_path, output_path):
    path_parts = input_path.rstrip("/").split("/")
    repo_id = path_parts[-2] + "_" + path_parts[-1]

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="robotwin_aloha_agilex",  # modify for your own robot
        fps=20,  # modify for your own robot
        features={
            "head_image": {
                "dtype": "image",
                "shape": (480, 640, 3),  # modify for your own data
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),  # modify for your own data
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),  # modify for your own data
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (14,),  # modify for your own data
                "names": [
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),  # modify for your own data
                "names": [
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                ],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def process_state(f):
    # 原始robotwin每一帧数据是数据里面: joint_action该时刻各关节运动到的位置, 对应了image;而不是根据image'要去执行的动作指令'。
    left_arm = f["joint_action/left_arm"][:-1]  # (105, 6)
    left_gripper = f["joint_action/left_gripper"][:-1].reshape(-1, 1)  # (105,1)
    right_arm = f["joint_action/right_arm"][:-1]  # (105, 6)
    right_gripper = f["joint_action/right_gripper"][:-1].reshape(-1, 1)  # (105,1)
    # 在列方向拼接
    return np.hstack([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float32)


def process_action(f):
    left_arm = f["joint_action/left_arm"][1:]
    left_gripper = f["joint_action/left_gripper"][1:].reshape(-1, 1)
    right_arm = f["joint_action/right_arm"][1:]
    right_gripper = f["joint_action/right_gripper"][1:].reshape(-1, 1)
    # 在列方向拼接
    return np.hstack([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float32)


def process_image(f):
    images_dict = {}
    for img_name in f["observation"]:
        num_frames = len(f["observation"][img_name]["rgb"])

        image_list = []
        # 转换每一帧图片格式
        for i in range(num_frames - 1):  # 最后一帧训练时候图像不用输入
            image_bit = f["observation"][img_name]["rgb"][i]
            image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (640, 480))
            # print(image.shape) # (480, 640, 3)
            image_list.append(image)
        image_array = np.array(image_list)
        images_dict[img_name] = image_array
    return images_dict


def process_instruction(episode_path):
    ep_num = int(os.path.basename(episode_path).replace("episode", "").replace(".hdf5", ""))
    instruction_dir = os.path.join(os.path.dirname(os.path.dirname(episode_path)), "instructions")
    instruction_type = "seen"
    instruction_path = os.path.join(instruction_dir, f"episode{ep_num}.json")
    with open(instruction_path) as f_instr:
        instruction_dict = json.load(f_instr)
        instructions = instruction_dict[instruction_type]
        instruction = random.choice(instructions)
        if not instruction.endswith("."):
            instruction += "."
    return instruction


def load_robotwin_ep_data(episode_path):
    with h5py.File(episode_path, "r") as f:
        # state =
        # state = torch.from_numpy(f["/observations/qpos"][:])
        states = torch.from_numpy(process_state(f))
        head_images = process_image(f)["head_camera"]
        left_wrist_images = process_image(f)["left_camera"]
        right_wrist_images = process_image(f)["right_camera"]
        actions = torch.from_numpy(process_action(f))

    instruction = process_instruction(episode_path)
    return states, head_images, left_wrist_images, right_wrist_images, actions, instruction


def convert_robotwin2lerobot(input_path, dataset):
    episode_dir = os.path.join(input_path, "data")
    episode_files = os.listdir(episode_dir)

    for i in range(len(episode_files)):
        episode_name = f"episode{i}.hdf5"
        episode_path = os.path.join(episode_dir, episode_name)
        print("处理文件:", episode_path)
        states, head_images, left_wrist_images, right_wrist_images, actions, instruction = load_robotwin_ep_data(
            episode_path
        )
        num_frames = len(states)
        for j in range(num_frames):
            frame = {
                "head_image": head_images[j],
                "left_wrist_image": left_wrist_images[j],
                "right_wrist_image": right_wrist_images[j],
                "state": states[j],
                "action": actions[j],
                "task": instruction,
            }
            dataset.add_frame(frame)
        dataset.save_episode()


def fix_parquet_metadata(parquet_path: pathlib.Path) -> None:
    """Fix the HuggingFace metadata in a parquet file by replacing 'List' with 'Sequence'."""
    try:
        # Read the parquet file
        parquet_file = pq.ParquetFile(parquet_path)

        # Get the schema with metadata
        schema = parquet_file.schema_arrow

        # Get and parse the HuggingFace metadata
        metadata = schema.metadata
        if b'huggingface' not in metadata:
            print(f"  未找到HuggingFace元数据: {parquet_path}")
            return

        hf_metadata = json.loads(metadata[b'huggingface'])

        # Fix the feature types: replace 'List' with 'Sequence'
        if 'info' in hf_metadata and 'features' in hf_metadata['info']:
            features = hf_metadata['info']['features']
            modified = False

            for key, feature in features.items():
                if isinstance(feature, dict) and feature.get('_type') == 'List':
                    print(f"    修复特征 '{key}': List -> Sequence")
                    feature['_type'] = 'Sequence'
                    modified = True

            if not modified:
                return

            # Update the metadata
            new_metadata = dict(metadata)
            new_metadata[b'huggingface'] = json.dumps(hf_metadata).encode('utf-8')

            # Create new schema with updated metadata
            new_schema = schema.with_metadata(new_metadata)

            # Read the table and write back with new schema
            table = parquet_file.read()
            table = table.cast(new_schema)

            # Write back to the same file
            pq.write_table(table, parquet_path)
            print(f"  ✓ 已修复: {parquet_path.name}")
    except Exception as e:
        print(f"  ✗ 修复失败 {parquet_path.name}: {e}")


def fix_all_parquet_files(dataset_path: str) -> None:
    """Fix all parquet files in the dataset directory."""
    dataset_dir = pathlib.Path(dataset_path)

    # Find all parquet files
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))

    if not parquet_files:
        print("未找到parquet文件")
        return

    print(f"\n修复parquet元数据 (共 {len(parquet_files)} 个文件)...")
    for parquet_file in parquet_files:
        fix_parquet_metadata(parquet_file)


def main(input_path, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    # os.makedirs(output_path, exist_ok=True)
    dataset = create_empty_lerobot(input_path, output_path)
    convert_robotwin2lerobot(input_path, dataset)

    # 自动修复parquet元数据
    fix_all_parquet_files(output_path)


if __name__ == "__main__":
    # robotwin data path # modify to your own path
    input_path = "/data/robotwin/robotwin_data/beat_block_hammer/aloha-agilex_demo-clean"

    # lerobot data path # modify to your own path
    output_path = "/data/robotwin/pi_data/beat_block_hammer/aloha-agilex_demo-clean"

    main(input_path, output_path)