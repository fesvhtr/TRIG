import base64
import json


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

