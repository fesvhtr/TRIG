from huggingface_hub import list_repo_files, hf_hub_download

from huggingface_hub import login



# 替换为你的 Hugging Face 访问令牌
token = "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"
login(token)

repo_id = "Yuanshi/Subjects200K"
# 列出仓库中的所有文件
files = list_repo_files(repo_id, repo_type="dataset")
# 筛选文件名中包含 "magicbrush" 的文件
# filtered_files = [f for f in files if "magicbrush" in f.lower()]  # 可以忽略大小写

for file in files:
    local_path = hf_hub_download(repo_id=repo_id, filename=file, cache_dir=r"H:\ProjectsPro\TRIG\dataset\raw_dataset\subject-driven", repo_type="dataset")
    print(f"Downloaded: {local_path}")
