# from datasets import load_dataset, Features, Value, Image
# # 1. 定义 schema：告诉 datasets 'image' 是 Image 类型，'prompt' 是 string
# features = Features({
#     "image": Image(),      # 这样 load_dataset 会自动读出 PIL.Image 而不是 path
#     "prompt": Value("string")
# })
# # 2. 载入并 cast
# ds = load_dataset(
#     "json",
#     data_files={"train": "/home/muzammal/Projects/TRIG/trig/ft/flux_ft_filtered.json"},
#     features=features,
#     split="train"
# )
# # 3. 保存到磁盘
# ds.save_to_disk("/home/muzammal/Projects/TRIG/trig/ft/flux_ft_ds")


from datasets import load_from_disk

# 1. 读入本地 dataset
ds = load_from_disk("/home/muzammal/Projects/TRIG/trig/ft/flux_ft_ds")

# 2. push 到你的 Hugging Face 空间，repository_id 格式为 "<你的用户名>/<数据集名>"
ds.push_to_hub("TRIG-bench/flux-ft-ds", private=False)
