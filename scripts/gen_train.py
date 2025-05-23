import json
import os
import sys
import openai
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trig.config import dim_definition
from tqdm import tqdm
import re
from typing import Any, List, Union

def extract_sentences(data: Any) -> List[str]:
    # 情况 1：字符串，匹配数字序号
    if isinstance(data, str):
        # (?s) = DOTALL 模式，让 . 能匹配换行
        pattern = r'(?s)\d+\.\s*(.+?)(?=(?:\n\d+\.|$))'
        matches = re.findall(pattern, data)
        sentences = [m.strip() for m in matches if m.strip()]
        if len(sentences) >= 3:
            return sentences[:3]
        else:
            raise ValueError("文本中未找到至少 3 条序号句子。")
    
    # 情况 2：列表或元组
    if isinstance(data, (list, tuple)):
        # 若本身就是字符串列表
        if all(isinstance(item, str) for item in data) and len(data) >= 3:
            return data[:3]
        # 若是元组且第二项是字符串列表
        if (isinstance(data, tuple) and
            len(data) >= 2 and
            isinstance(data[1], list) and
            all(isinstance(item, str) for item in data[1]) and
            len(data[1]) >= 3):
            return data[1][:3]
    
    raise ValueError(f"不支持的输入类型: {type(data)}")

API_KEY = "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"
client = openai.Client(api_key=API_KEY)

with open("/home/muzammal/Projects/TRIG/dataset/Trig/Trig-text-to-image/text-to-imgae-without-m-e-v8.json", "r") as f:
    ori_data = json.load(f)

ref_data = []
new_data = []

for i in ori_data:
    if i['dimension'] == ["D-K", "D-A"] or i['dimension'] == ["TA-R", "TA-S"] or i['dimension'] == ["IQ-R", "TA-C"]:
        ref_data.append(i)
print("ref_data:", len(ref_data))

for i in tqdm(ref_data[166:]):
    prompt = i["prompt"]
    dim = i["dimension"]
    # print("prompt:",prompt)
    # print("dim:",dim)
    sys_msg = [{
                "role": "developer",
                "content": """
                You're a data generation assistant, and I'm going to give you some data afterward, each of which is a prompt for image generation, and each of which focuses very obviously on two assessment dimensions.
                I want you to augment this data so that I have more data, which means that you need to use this prompt I gave you as a basis for finding out what is in the prompt about the 
                two dimensions by using the names and definitions of the two dimensions that I gave you, and then change those so that it becomes a new prompt.

                You need to do the following:
                1. Understand what these two dimensions are, they are the fixed dimensions that I gave you!!!!!!! Don't find new dimensions, just use the two dimensions I gave you.
                2. Understand how this prompt relates to these two dimensions and what are the features
                3. Change the prompt so that it is a new prompt but still related to the two dimensions
                4. Make sure that the new prompt is not too similar to the original prompt, but still strongly related to the two dimensions
                5. If the original prompt doesn't explicitly relate to both dimensions, you'll want to add information so that it can include both dimensions.

                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                It's very important that you note that you're helping me increase the quantity of data, not the quality, and that you just generate the exact same level of data.
                In particular, don't increase the length, make sure the PROMPT a midium length (less than 50 words) sentence that fulfills both dimensions.
                don't increase the length! (less than 50 words) 
                Remember that the prompt is a sentence that fulfills both dimensions, only give me the new prompt, and don't give me any other information.

                You need give me three new prompts, and each of them should be different from the original prompt and from each other!!!
                You need to change the main content of the prompt, don't just change the expression of the prompt, but also change the content of the prompt.

                Give me three prompts in direct order and label them from one to three
                You don't need to give me dimensions, it's fixed, you don't need to think about the two dimensions on your own
                Just give me the new prompts, and don't give me any other information. in this format:
                1. <prompt1>
                2. <prompt2>
                3. <prompt3>


                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                Don't give me any other information, just give me the new prompts.
                """
            }]
    user_msg_content = """
    prompt: {}
    The two dimensions are:
    1. {}
    2. {}
    Directly change the prompt to 3 new ones but still strongly related to the two dimensions.

    """.format(prompt, dim_definition[dim[0]], dim_definition[dim[1]])
    print("user_msg_content:", user_msg_content)
    user_msg = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},]
            }]
    try:
        # 尝试请求 API
        completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=sys_msg + user_msg,
                        timeout=60  # 设置超时时间
                    )
        new_prompt = completion.choices[0].message.content.strip()
        
        # 提取新 prompts
        new_prompts = extract_sentences(new_prompt)
        if not new_prompts or len(new_prompts) != 3:
            print(f"提取失败: {new_prompt}")
            continue
        
        # 过滤空字符串
        with open("/home/muzammal/Projects/TRIG/dataset/Trig/Trig-text-to-image/text-to-image-ft.json", "a") as f:
            for new_prompt_ in new_prompts:
                # 去重并过滤空字符串
                new_prompt_ = new_prompt_.strip()
                if new_prompt_:
                    data_item = {"prompt": new_prompt_, "dimension": dim}
                    new_data.append(data_item)
                    f.write(json.dumps(data_item, ensure_ascii=False) + "\n")


    except Exception as e:
        print(f"未知错误: {e}")
        continue


with open("/home/muzammal/Projects/TRIG/dataset/Trig/Trig-text-to-image/text-to-image-ft.json", "w") as f:
    json.dump(new_data, f, indent=4)
