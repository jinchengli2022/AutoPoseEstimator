#!/bin/bash

# 设置默认序号，默认从1开始
sequence=1

# 如果传入了参数，则使用传入的序号
if [ $# -gt 0 ]; then
  sequence=$1
fi

# 设置目标文件夹路径（可以根据需要修改）
folder_path="Data/real_data/pour_water/episode_${sequence}"  # 修改为你想要压缩的文件夹路径
output_dir="Output"        # 压缩文件保存的目录

# 检查文件夹是否存在
if [ ! -d "$folder_path" ]; then
  echo "错误: 文件夹路径 $folder_path 不存在!"
  exit 1
fi

# 创建输出目录，如果没有的话
mkdir -p "$output_dir"

# 压缩文件夹并生成带有序号的文件名
output_file="${output_dir}/episode_${sequence}.tar.gz"

# 执行压缩
tar -czf "$output_file" -C "$(dirname "$folder_path")" "$(basename "$folder_path")"

# 输出完成信息
echo "压缩完成: $output_file"
