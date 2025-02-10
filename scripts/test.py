
import numpy as np
import scipy.stats as stats
import pandas as pd

# 示例数据：两个度量列表（用实际数据替换这些示例数据）
list1 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # 度量1
list2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 度量2

# 计算皮尔逊相关系数
pearson_corr, _ = stats.pearsonr(list1, list2)

# 计算斯皮尔曼相关系数
spearman_corr, _ = stats.spearmanr(list1, list2)

# 创建一个数据框，方便显示相关系数
correlation_data = {
    "度量1": list1,
    "度量2": list2,
    "皮尔逊相关系数": [pearson_corr] * len(list1),
    "斯皮尔曼相关系数": [spearman_corr] * len(list1),
}

df = pd.DataFrame(correlation_data)

import ace_tools as tools; tools.display_dataframe_to_user(name="相关性分析结果", dataframe=df)

# 返回相关系数
print(f"皮尔逊相关系数: {pearson_corr}")
print(f"斯皮尔曼相关系数: {spearman_corr}")