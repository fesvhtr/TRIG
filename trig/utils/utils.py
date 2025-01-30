import base64
import yaml
from PIL import Image
from io import BytesIO

def get_image_type(image_path):
    image_type = image_path.split('.')[-1]
    assert image_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']
    return image_type


def encode_image(image_path):
    image = {}
    with open(image_path, "rb") as image_file:
        image['base64'] = base64.b64encode(image_file.read()).decode('utf-8')
    image['type'] = get_image_type(image_path)
    return image

def load_config( config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def base64_to_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image_buffer = BytesIO(image_data)
        image = Image.open(image_buffer)
        return image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None