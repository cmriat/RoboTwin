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


def create_empty_lerobot(input_path, output_path, mode="video"):
    path_parts = input_path.rstrip("/").split("/")
    repo_id = path_parts[-2] + "_" + path_parts[-1]

    motors = [
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
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="robotwin_aloha_agilex",
        fps=20,
        features=features,
        use_videos=(mode == "video"),
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
                "observation.images.cam_high": head_images[j],
                "observation.images.cam_left_wrist": left_wrist_images[j],
                "observation.images.cam_right_wrist": right_wrist_images[j],
                "observation.state": states[j],
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


def create_modality_json(output_path: str) -> None:
    """Create modality.json file for the dataset."""
    meta_dir = os.path.join(output_path, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    modality = {
        "state": {
            "left_arm": {"start": 0, "end": 6},
            "left_gripper": {"start": 6, "end": 7},
            "right_arm": {"start": 7, "end": 13},
            "right_gripper": {"start": 13, "end": 14}
        },
        "action": {
            "left_arm": {"start": 0, "end": 6},
            "left_gripper": {"start": 6, "end": 7},
            "right_arm": {"start": 7, "end": 13},
            "right_gripper": {"start": 13, "end": 14}
        },
        "video": {
            "cam_high": {"original_key": "observation.images.cam_high"},
            "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"}
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
            "human.validity": {},
            "human.coarse_action": {}
        }
    }

    modality_path = os.path.join(meta_dir, "modality.json")
    with open(modality_path, "w") as f:
        json.dump(modality, f, indent=4)
    print(f"  ✓ 已创建: {modality_path}")


def main(input_path, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    # os.makedirs(output_path, exist_ok=True)
    dataset = create_empty_lerobot(input_path, output_path)
    convert_robotwin2lerobot(input_path, dataset)

    # 自动修复parquet元数据
    fix_all_parquet_files(output_path)

    # 创建modality.json
    create_modality_json(output_path)


if __name__ == "__main__":
    # 处理所有数据集
    base_input_path = "/data/robotwin/robotwin_data"
    base_output_path = "/data/robotwin/groot_data_new"
    
    # 获取所有数据集目录
    dataset_dirs = [d for d in os.listdir(base_input_path) 
                   if os.path.isdir(os.path.join(base_input_path, d))]
    
    print(f"找到 {len(dataset_dirs)} 个数据集目录: {dataset_dirs}")
    
    for dataset_name in dataset_dirs:
        print(f"\n开始处理数据集: {dataset_name}")
        
        # 构建输入和输出路径
        input_path = os.path.join(base_input_path, dataset_name, "aloha-agilex_demo-clean")
        output_path = os.path.join(base_output_path, dataset_name, "aloha-agilex_demo-clean")
        
        # 检查输入路径是否存在
        if not os.path.exists(input_path):
            print(f"  警告: 输入路径不存在，跳过: {input_path}")
            continue
            
        try:
            print(f"  输入路径: {input_path}")
            print(f"  输出路径: {output_path}")
            
            # 处理单个数据集
            main(input_path, output_path)
            print(f"  ✓ 数据集 {dataset_name} 处理完成")
            
        except Exception as e:
            print(f"  ✗ 数据集 {dataset_name} 处理失败: {e}")
            print("程序终止！")
            exit(1)
    
    print(f"\n所有数据集处理完成！")