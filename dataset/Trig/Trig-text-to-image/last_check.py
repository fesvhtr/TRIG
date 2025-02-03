import json
import random
with open(r'H:\ProjectsPro\TRIG\dataset\Trig\Trig-text-to-image\text-to-imgae.json', 'r') as f:
    data = json.load(f)

docci = []
with open(r"H:\ProjectsPro\TRIG\dataset\Trig\Trig-text-to-image\docci_descriptions.jsonlines", 'r', encoding='utf-8') as f:
    for line in f:
        # 解析每一行的 JSON 对象
        docci.append(json.loads(line.strip()))


for i in data:
    if i['dimension'] == ["IQ-R","TA-R"]:
        id = i["data_id"].split("_")[-1]
        if int(id) > 265:
            replace_data = random.choice(docci)
            i['prompt'] = replace_data['description']
            i['dimension_prompt'] = ["","Complex spatial relation"]
            i['img_id'] = replace_data['image_file']
            i['parent_dataset'] = ["DOCCI","Origin"]

with open(r'H:\ProjectsPro\TRIG\dataset\Trig\Trig-text-to-image\text-to-imgae-new1.json', 'w') as f:
    json.dump(data, f, indent=4)

