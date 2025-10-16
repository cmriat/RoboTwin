torchrun --standalone --nproc_per_node=8 scripts/train_pytorch_fsdp_robotwin.py \
      --exp_name robotwin \
      --shuffle \
      --num-workers 8 \
      --batch-size 32 \
      --train-steps 30000
# 1. GPU数量: --nproc_per_node 应该等于可用GPU数量
# 2. Batch size: 必须能被GPU数量整除（代码第241行检查）
# 3. 数据集: 确保已经运行 scripts/compute_norm_stats.py 生成归一化统计数据
# 4. checkpoint会保存到 /data/robot/checkpoints/pi05/{exp_name}/



###
# 2. 单机单GPU训练 (测试用)

#   torchrun --standalone --nproc_per_node=1 scripts/train_pytorch_fsdp_robotwin.py \
#       --exp_name test_run \
#       --shuffle \
#       --batch-size 32 \
#       --train-steps 1000

#   3. 从多个数据集目录加载

#   torchrun --standalone --nproc_per_node=4 scripts/train_pytorch_fsdp_robotwin.py \
#       --exp_name multi_dataset \
#       --data-root /path/to/datasets \
#       --shuffle \
#       --batch-size 64 \
#       --train-steps 30000