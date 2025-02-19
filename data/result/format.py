import json
from collections import defaultdict

def simplify_json(input_file, output_file):
    # 读取原始JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 创建新的格式化数据
    formatted_data = defaultdict(dict)
    
    # 遍历所有数据项
    for key in data:
        # 获取维度对前缀（如 "D-K_D-A"）和序号
        prefix, number = key.rsplit('_', 1)
        
        # 提取两个指标的值
        values = []
        for metric in data[key].values():
            value = metric[0]["TRIGAPIMetric_Benasd/Qwen2.5-VL-72B-Instruct-AWQ"]
            values.append(value)
        
        # 添加到对应维度对的字典中
        formatted_data[prefix][f"{prefix}_{number}"] = values
    
    # 转换defaultdict为普通dict
    formatted_data = dict(formatted_data)
    
    # 保存为新的JSON文件
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)

# 使用示例
input_file = 'data/result/flux_72B_t1.json'  # 原始JSON文件路径
output_file = 'data/result/formatted_flux_72B_t1.json'  # 输出文件路径
simplify_json(input_file, output_file) 