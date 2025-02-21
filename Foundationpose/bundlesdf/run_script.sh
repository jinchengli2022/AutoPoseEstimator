#!/bin/bash

# 检查是否以 root 权限运行
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo."
  exit 1
fi

# 定义变量
PYTHON_PATH="/home/ljc/Git/FoundationPose/bundlesdf/run_nerf.py"
REF_VIEW_DIR="my_linemod/cola_solve"
DATASET="linemod"

# 执行命令
python "$PYTHON_PATH" --ref_view_dir "$REF_VIEW_DIR" --dataset "$DATASET"
