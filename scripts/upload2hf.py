from huggingface_hub import HfApi

# api = HfApi()
# api.upload_file(
#     path_in_repo="output/t2i/omnigen.zip", 
#     path_or_fileobj="/home/muzammal/Projects/TRIG/data/output/t2i/sd35.zip",  
#     repo_id="TRIG-bench/TRIG", 
#     repo_type="dataset" 
# )

from huggingface_hub import hf_hub_download

# 设定仓库ID
repo_id = "TRIG-bench/TRIG"

# 远程仓库中的文件路径
path_in_repo = "dataset/Trig-subject-driven/images.zip"

# 指定本地存储路径（可选）
local_file = hf_hub_download(
    repo_id=repo_id, 
    filename=path_in_repo, 
    repo_type="dataset",
    local_dir="/home/muzammal/Projects/TRIG/dataset/Trig"
)

print(f"文件已下载到: {local_file}")
