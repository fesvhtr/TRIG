import os
import re
import base64
import imghdr
import json
from openai import OpenAI
from tqdm import tqdm


DIM_DICT = {
    "IQ-R": ["IQ-O", "IQ-A", "TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "IQ-O": ["IQ-A", "TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "IQ-A": ["TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-C": ["TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-R": ["TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-S": ["D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "D-M":  ["D-K", "D-A", "R-T", "R-B", "R-E"],
    "D-K":  ["D-A", "R-T", "R-B", "R-E"],
    "D-A":  ["R-T", "R-B", "R-E"],
    "R-T":  ["R-B", "R-E"],
    "R-B":  ["R-E"]
}
DIM_DESC = {
    "IQ-R": "Image Quality, Realism, Similarity between the generated images and those in the real world",
    "IQ-O": "Image Quality, Originality, Novelty and uniqueness in the generated images.",
    "IQ-A": "Image Quality, Aesthetics, Aesthetic level of the generated images for people visually.",
    "TA-C": "Task Alignment, Content Alignment, Alignment of the image's main objects and scenes with those specified in the prompt.",
    "TA-R": "Task Alignment, Relation Alignment, Alignment of the image's spatial and semantic logical relationships between humans and objects with those specified in the prompt.",
    "TA-S": "Task Alignment, Style Alignment, Alignment of the image's style (scheme and aesthetic) with that specified in the prompt.",
    "D-M":  "Diversity, Multilingual, Ability to generate images based on prompts provided in multiple languages.",
    "D-K":  "Diversity, Knowledge, Ability to generate images with complex or specialized knowledge.",
    "D-A":  "Diversity, Ambiguous, Ability to generate images based on prompts that are ambiguous or abstract.",
    "R-T":  "Robustness, Toxicity, The extent to which the generated images contain harmful or offensive content.",
    "R-B":  "Robustness, Bias, The extent to which the generated images exhibit biases.",
    "R-E":  "Robustness, Efficiency, Computational resources and time required by the model in the generation process."
}


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def encode_image(image_path):
    """Encodes an image to Base64 and detects its type."""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_type = imghdr.what(None, image_data)  # Detect image type (e.g., 'jpeg', 'png')
        if image_type not in ["jpeg", "png"]:
            raise ValueError(f"Unsupported image format: {image_type}")
        base64_image = base64.b64encode(image_data).decode("utf-8")
        return base64_image, image_type
    

def create_message(base64_image, image_type, dim1_desc, dim2_desc):
    
    edit_message = {
        "role": "system",
        "content": (
            "You are an AI vision expert. Your task is to analyze image content and generate prompts for image editing.\n\n"
            "**`Response Constraints`**:\n"
            "- The output must be in JSON format containing two elements:\n"
            "  - `<prompt>`: A string describing the image editing task.\n"
            "  - `<dimension_prompt>`: A list of two strings, each representing key information for two target dimensions.\n"
            "- The response should contain only the required prompt information, without any additional explanations.\n\n"
            "**`Response Requirements`**:\n"
            "- The `<prompt>` should be generated based on the given image content.\n"
            "- The `<prompt>` must include descriptions for `two dimensions`:\n"
            f"  - Dimension 1: {dim1_desc}\n"
            f"  - Dimension 2: {dim2_desc}\n"
            "- The `<dimension_prompt>` should be extracted directly from the `<prompt>` as concise phrases representing each dimension.\n"
            "- The order of `<dimension_prompt>` must match the order of the provided dimensions (`dim1` first, then `dim2`)."
        ),
    }

    src_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze the given image and generate a JSON output as per the specified structure."},
            {"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{base64_image}"}}
        ]
    }

    message = [edit_message, src_message]
    return message



def send_request(client, messages):
    """Sends a request to the OpenAI API with the given messages."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content


def save_results(dim, img_id, response_content):
    json_path = './prompts'
    os.makedirs(json_path, exist_ok=True)
    json_file = os.path.join(json_path, f'{dim}.json')
    
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            dim_file = json.load(file)
            
        if img_id == dim_file[-1]["img_id"]:
            last_data_id = int(dim_file[-1]["data_id"].split("_")[-1])
            data_id = f"{dim}_{last_data_id + 1}"
        else:
            return
    else:
        dim_file = []
        data_id = f"{dim}_1"
    
    meta_data = {
        'data_id': data_id,
        'prompt': response_content['prompt'],
        'dimension_prompt': response_content['dimension_prompt'],
        'img_id': img_id,
        "parent_dataset": [
            "OmniEdit-Filtered"
            "Origin"
        ]
    }
    dim_file.append(meta_data)
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(dim_file, file, ensure_ascii=False, indent=4)


def process_data(client, data_list, raw_path):
    for object in tqdm(data_list, desc="Processing object", unit="image"):
        src_img_filename = object['src_img_filename']
        image_path = os.path.join(raw_path, src_img_filename)
        base64_image, image_type = encode_image(image_path)

        for dim1, dim2_list in DIM_DICT.items():
            for dim2 in dim2_list:
                dim1_desc = DIM_DESC[dim1]
                dim2_desc = DIM_DESC[dim2]
                
                image_message = create_message(base64_image, image_type, dim1_desc, dim2_desc)
                response_content = send_request(client, image_message)
                response_match = re.search(r"\{.*\}", response_content, re.DOTALL)
                response_content = json.loads(response_match.group(0))
                
                dim = f'{dim1}_{dim2}'
                img_id = src_img_filename
                save_results(dim, img_id, response_content)
            

if __name__ == "__main__":
    api_key = "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"
    client = OpenAI(api_key=api_key)    
    
    raw_path = '../../raw_datasets/subject-driven/Subjects200K-Total'
    for file_name in ['collection1']:
        file_path = os.path.join(raw_path, file_name, f'{file_name}.json')
        data_list = load_json(file_path)
        process_data(client, data_list, raw_path)
