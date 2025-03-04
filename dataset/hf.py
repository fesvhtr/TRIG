from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm
from huggingface_hub import login
import os
import zipfile

def zip_subfolders(parent_folder):
    """
    遍历 parent_folder 下所有子文件夹，将每个子文件夹打包成一个对应的 zip 文件。
    
    参数:
        parent_folder: 包含多个子文件夹的父文件夹路径
    """
    # 检查父文件夹是否存在
    if not os.path.exists(parent_folder):
        raise ValueError(f"指定的文件夹 {parent_folder} 不存在")
    
    # 遍历父文件夹下的所有条目
    for item in tqdm(os.listdir(parent_folder)):
        item_path = os.path.join(parent_folder, item)
        # 如果该条目是文件夹，则进行打包
        if os.path.isdir(item_path):
            zip_file_path = os.path.join(parent_folder, f"{item}.zip")
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 遍历该子文件夹的所有文件和子文件夹
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 计算相对路径，这样压缩包内会保留目录结构
                        # 这里 arcname 以子文件夹名称为根目录
                        arcname = os.path.join(item, os.path.relpath(file_path, item_path))
                        zipf.write(file_path, arcname)
            print(f"已将文件夹 {item_path} 压缩为 {zip_file_path}")

# 示例用法
if __name__ == "__main__":
    parent_dir = "/home/muzammal/Projects/TRIG/data/output/p2p"  # 修改为你的文件夹路径
    zip_subfolders(parent_dir)



# # 替换为你的 Hugging Face 访问令牌
# token = "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"
# login(token)

# repo_id = "yzwang/X2I-subject-driven"
# # 列出仓库中的所有文件
# files = list_repo_files(repo_id, repo_type="dataset")
# # 筛选文件名中包含 "magicbrush" 的文件
# filtered_files = [f for f in files if "character" in f.lower()]  # 可以忽略大小写

# for file in filtered_files:
#     local_path = hf_hub_download(repo_id=repo_id, filename=file, cache_dir=r"/home/muzammal/Projects/TRIG/dataset", repo_type="dataset")
#     print(f"Downloaded: {local_path}")
