import base64
import yaml
from PIL import Image
from io import BytesIO


def get_image_resolution(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def get_image_type(image_path):
    image_type = image_path.split('.')[-1]
    assert image_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']
    return image_type


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def image_data_url(image_input, max_side=None):
    """Data URL for an image path or PIL image, optionally downscaled to ``max_side`` pixels."""
    if max_side:
        img = Image.open(image_input) if isinstance(image_input, str) else image_input
        if max(img.size) > max_side:
            img = img.copy()
            img.thumbnail((max_side, max_side))
            image_input = img
    enc = encode_image(image_input)
    return f"data:image/{enc['type']};base64,{enc['base64']}"


def encode_image(image_input):
    image = {}
    
    # Detect input type: file path string or PIL Image object
    if isinstance(image_input, str):
        # Handle file path
        with open(image_input, "rb") as image_file:
            image['base64'] = base64.b64encode(image_file.read()).decode('utf-8')
        image['type'] = get_image_type(image_input)
    elif isinstance(image_input, Image.Image):
        # Handle PIL Image object
        buffer = BytesIO()
        # Get image format, default to PNG if not available
        image_format = image_input.format if image_input.format else 'PNG'
        # Save to byte stream
        image_input.save(buffer, format=image_format)
        buffer.seek(0)
        # Encode to base64
        image['base64'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # Set type
        image['type'] = image_format.lower()
    else:
        raise ValueError("image_input must be either a file path (str) or a PIL Image object")
    
    return image


def base64_to_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image_buffer = BytesIO(image_data)
        image = Image.open(image_buffer)
        return image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

