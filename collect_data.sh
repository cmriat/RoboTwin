#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

# 从配置文件读取 save_path
save_path=$(grep "^save_path:" task_config/${task_config}.yml | awk '{print $2}')

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
rm -rf ${save_path}/${task_name}/${task_config}/.cache

# 删除 _traj_data 目录（中间缓存，采集完成后不再需要）
rm -rf ${save_path}/${task_name}/${task_config}/_traj_data
echo "Deleted _traj_data directory"

# 只保留前 20 个视频文件
video_dir=${save_path}/${task_name}/${task_config}/video
if [ -d "$video_dir" ]; then
    ls -1 "$video_dir"/episode*.mp4 2>/dev/null | sort -t 'e' -k2 -n | tail -n +21 | xargs -r rm -f
    echo "Kept only first 20 videos in $video_dir"
fi


# bash collect_data.sh adjust_bottle aloha-agilex_demo-clean 0