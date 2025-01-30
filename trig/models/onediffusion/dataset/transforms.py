import torch
import torch.nn.functional as F

def crop(image, i, j, h, w):
    """
    Args:
        image (torch.tensor): Image to be cropped. Size is (C, H, W)
    """
    if len(image.size()) != 3:
        raise ValueError("image should be a 3D tensor")
    return image[..., i : i + h, j : j + w]

def resize(image, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return F.interpolate(image.unsqueeze(0), size=target_size, mode=interpolation_mode, align_corners=False).squeeze(0)

def resize_scale(image, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    H, W = image.size(-2), image.size(-1)
    scale_ = target_size[0] / min(H, W)
    return F.interpolate(image.unsqueeze(0), scale_factor=scale_, mode=interpolation_mode, align_corners=False).squeeze(0)

def resized_crop(image, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the image
    Args:
        image (torch.tensor): Image to be cropped. Size is (C, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized image
    Returns:
        image (torch.tensor): Resized and cropped image. Size is (C, H, W)
    """
    if len(image.size()) != 3:
        raise ValueError("image should be a 3D torch.tensor")
    image = crop(image, i, j, h, w)
    image = resize(image, size, interpolation_mode)
    return image

def center_crop(image, crop_size):
    if len(image.size()) != 3:
        raise ValueError("image should be a 3D torch.tensor")
    h, w = image.size(-2), image.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(image, i, j, th, tw)

def center_crop_using_short_edge(image):
    if len(image.size()) != 3:
        raise ValueError("image should be a 3D torch.tensor")
    h, w = image.size(-2), image.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(image, i, j, th, tw)

class CenterCropResizeImage:
    """
    Resize the image while maintaining aspect ratio, and then crop it to the desired size.
    The resizing is done such that the area of padding/cropping is minimized.
    """
    def __init__(self, size, interpolation_mode="bilinear"):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"Size should be a tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation_mode = interpolation_mode

    def __call__(self, image):
        """
        Args:
            image (torch.Tensor): Image to be resized and cropped. Size is (C, H, W)
        
        Returns:
            torch.Tensor: Resized and cropped image. Size is (C, target_height, target_width)
        """
        target_height, target_width = self.size
        target_aspect = target_width / target_height

        # Get current image shape and aspect ratio
        _, height, width = image.shape
        height, width = float(height), float(width)
        current_aspect = width / height

        # Calculate crop dimensions
        if current_aspect > target_aspect:
            # Image is wider than target, crop width
            crop_height = height
            crop_width = height * target_aspect
        else:
            # Image is taller than target, crop height
            crop_height = width / target_aspect
            crop_width = width

        # Calculate crop coordinates (center crop)
        y1 = (height - crop_height) / 2
        x1 = (width - crop_width) / 2

        # Perform the crop
        cropped_image = crop(image, int(y1), int(x1), int(crop_height), int(crop_width))

        # Resize the cropped image to the target size
        resized_image = resize(cropped_image, self.size, self.interpolation_mode)

        return resized_image
    
# Example usage
if __name__ == "__main__":
    # Create a sample image tensor
    sample_image = torch.rand(3, 480, 640)  # (C, H, W)

    # Initialize the transform
    transform = CenterCropResizeImage(size=(224, 224), interpolation_mode="bilinear")

    # Apply the transform
    transformed_image = transform(sample_image)

    print(f"Original image shape: {sample_image.shape}")
    print(f"Transformed image shape: {transformed_image.shape}")