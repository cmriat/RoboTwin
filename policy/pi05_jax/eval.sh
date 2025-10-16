#!/bin/bash

policy_name=pi05_jax # [TODO] 
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${policy_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 
    

# bash eval.sh adjust_bottle 50ep-agilex-demo_clean pi05_robotwin /data/robot/checkpoints/pi05_jax/pi05_robotwin/robotwin_jax 0 0
# --overrides 后面我在命令行里输入的参数，要覆盖配置文件（config）里已有的默认设置。
# train_config_name 官方openpi里面policy/config.py里面的TrainConfig其中一个名字，这里我需要自己改成pi05_robotwin