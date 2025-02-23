import gdown
import zipfile
import tarfile
import os
import time

# 设置 Google Drive 文件的文件ID
# https://drive.google.com/file/d/1o3Ci3YGtHTcOxdSuCwcJhToWRJIq6kNr/view?usp=drive_link
file_id = '1o3Ci3YGtHTcOxdSuCwcJhToWRJIq6kNr'

# 设置下载文件保存的路径
downloaded_file = 'weights.tar.xz'
output_dir = 'weights'

# 使用 gdown 下载文件
url = f'https://drive.google.com/uc?export=download&id={file_id}'
gdown.download(url, downloaded_file, quiet=False)

# 等待一段时间，确保文件完全下载（如果需要）
time.sleep(5)  # 等待 5 秒钟，可以根据需要调整

# 检查文件是否存在
if os.path.exists(downloaded_file):
    try:
        # 解压 .tar.xz 文件
        with tarfile.open(downloaded_file, 'r:xz') as tar_ref:
            tar_ref.extractall(output_dir)  # 提取到 'weights' 目录
        print(f"文件已成功解压到 '{output_dir}' 目录。")
    except tarfile.TarError:
        print("下载的文件不是一个有效的 .tar.xz 文件。")
else:
    print("文件不存在，请检查下载路径。")
