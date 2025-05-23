import json
from collections import defaultdict

# 输入和输出文件路径
input_file = "/home/muzammal/Projects/TRIG/dataset/Trig/Trig-text-to-image/text-to-image-ft.json"
output_file = "/home/muzammal/Projects/TRIG/dataset/Trig/Trig-text-to-image/text-to-image-ft-v1.json"

# 读取原始数据
with open(input_file, "r") as f:
    data = json.load(f)

# 维度组合计数
dim_counters = defaultdict(int)

# 更新每条数据
for item in data:
    # 取出两个维度
    dim1, dim2 = item["dimension"]
    # 组合 ID 前缀
    prefix = f"{dim1}_{dim2}"
    # 计数递增
    dim_counters[prefix] += 1
    # 设置 id
    item["data_id"] = f"{prefix}_{dim_counters[prefix]}"
    # 初始化 img_id
    item["img_id"] = ""
    item["dimension_prompt"] = ["",""]
    item["parent_dataset"] = ["","Auto"]

# 保存更新后的数据
with open(output_file, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ 处理完成，共生成 {len(data)} 条数据")
