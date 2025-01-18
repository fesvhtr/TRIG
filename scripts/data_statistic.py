import json
from collections import defaultdict
import pandas as pd


def convert_result_to_df(result):
    data_for_df = []
    total_dimension_prompt_counts = [0, 0]
    for class_name, stats in result.items():
        total_dimension_prompt_counts[0] += stats["dimension_prompt"][0]
        total_dimension_prompt_counts[1] += stats["dimension_prompt"][1]
        data_for_df.append({
            "Class": class_name,
            "Count": int(stats["count"]),
            "Dimension Prompt": stats["dimension_prompt"],
            "Dimensions": stats["dimensions"],
            "Images": stats["img_id"],
            "Parent Dataset": list(set(stats["parent_dataset"])),
            "Annotation": list(set(stats["annotation"]))
        })
    df = pd.DataFrame(data_for_df)
    total_count = df["Count"].sum()
    total_images = df["Images"].sum()
    total_row = {
        "Class": "Total",
        "Count": total_count,
        "Dimension Prompt": total_dimension_prompt_counts,
        "Dimensions": "---",
        "Images": total_images,
        "Parent Dataset": "---",
        "Annotation": "---"
    }
    df = df.append(total_row, ignore_index=True)
    return df


def data_statistic(data_list):
    result = defaultdict(lambda: {
        "count": 0,
        "dimension_prompt": [0, 0],
        "dimension": ['', ''],
        "img_id": 0,
        "parent_dataset": [],
        'annotation': []
    })

    for data in data_list:
        class_name = "_".join(data["data_id"].split("_")[:-1])

        result[class_name]["count"] += 1

        dimension_prompt = data["dimension_prompt"]
        if dimension_prompt[0]:
            result[class_name]["dimension_prompt"][0] += 1
        if dimension_prompt[1]:
            result[class_name]["dimension_prompt"][1] += 1

        result[class_name]['dimensions'] = [class_name.split('_')[0], class_name.split('_')[1]]

        img_id = data['img_id']
        if img_id:
            result[class_name]['img_id'] += 1
        result[class_name]["parent_dataset"].append(data["parent_dataset"][0])
        result[class_name]["annotation"].append(data["parent_dataset"][1])

    # 返回统计结果
    return result


if __name__ == "__main__":
    subsets = ['t2i', 'subject_gen']
    with open(f'../data/dataset/prompts/trim_t2i.json', 'r') as f:
        data_list = json.load(f)
    df = convert_result_to_df(data_statistic(data_list))
    print(df.to_string())
