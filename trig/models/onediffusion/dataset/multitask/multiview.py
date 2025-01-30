import os
import json
import random
from PIL import Image
import torch
from typing import List, Tuple, Union
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T
from onediffusion.dataset.utils import *
import glob

from onediffusion.dataset.raydiff_utils import cameras_to_rays, first_camera_transform, normalize_cameras
from onediffusion.dataset.transforms import CenterCropResizeImage
from pytorch3d.renderer import PerspectiveCameras

import numpy as np

def _cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
    do_normalize_cameras,
    normalize_scale,
) -> PerspectiveCameras:
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    cams = PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=R.device,
    )
    
    if do_normalize_cameras:
        cams, _ = normalize_cameras(cams, scale=normalize_scale)
    
    cams = first_camera_transform(cams, rotation_only=False)
    return cams

def calculate_rays(Ks, sizes, Rs, Ts, target_size, use_plucker=True, do_normalize_cameras=False, normalize_scale=1.0):
    cameras = _cameras_from_opencv_projection(
        R=Rs,
        tvec=Ts,
        camera_matrix=Ks,
        image_size=sizes,
        do_normalize_cameras=do_normalize_cameras,
        normalize_scale=normalize_scale
    )
        
    rays_embedding = cameras_to_rays(
        cameras=cameras,
        num_patches_x=target_size,
        num_patches_y=target_size,
        crop_parameters=None,
        use_plucker=use_plucker
    )
        
    return rays_embedding.rays

def convert_rgba_to_rgb_white_bg(image):
    """Convert RGBA image to RGB with white background"""
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        # Composite the image onto the white background
        return Image.alpha_composite(background, image).convert('RGB')
    return image.convert('RGB')

class MultiviewDataset(Dataset):
    def __init__(
        self, 
        scene_folders: str, 
        samples_per_set: Union[int, Tuple[int, int]],  # Changed from samples_per_set to samples_range
        transform=None, 
        caption_keys: Union[str, List] = "caption",
        multiscale=False, 
        aspect_ratio_type=ASPECT_RATIO_512,
        c2w_scaling=1.7,
        default_max_distance=1, # default max distance from all camera of a scene ,
        do_normalize=True, # whether normalize translation of c2w with max_distance
        swap_xz=False, # whether swap x and z axis of 3D scenes
        valid_paths: str = "",
        frame_sliding_windows: float = None # limit all sampled frames to be within this window, so that camera poses won't be too different
    ):
        if not isinstance(samples_per_set, tuple) and not isinstance(samples_per_set, list):
            samples_per_set = (samples_per_set, samples_per_set)
        self.samples_range = samples_per_set  # Tuple of (min_samples, max_samples)
        self.transform = transform
        self.caption_keys = caption_keys if isinstance(caption_keys, list) else [caption_keys]
        self.aspect_ratio = aspect_ratio_type
        self.scene_folders = sorted(glob.glob(scene_folders))
        # filter out scene folders that do not have transforms.json
        self.scene_folders = list(filter(lambda x: os.path.exists(os.path.join(x, "transforms.json")), self.scene_folders))

        # if valid_paths.txt exists, only use paths in that file
        if os.path.exists(valid_paths):
            with open(valid_paths, 'r') as f:
                valid_scene_folders = f.read().splitlines()
            self.scene_folders = sorted(valid_scene_folders)
            
        self.c2w_scaling = c2w_scaling
        self.do_normalize = do_normalize
        self.default_max_distance = default_max_distance
        self.swap_xz = swap_xz
        self.frame_sliding_windows = frame_sliding_windows
        
        if multiscale:
            assert self.aspect_ratio in [ASPECT_RATIO_512, ASPECT_RATIO_1024, ASPECT_RATIO_2048, ASPECT_RATIO_2880]
            if self.aspect_ratio in [ASPECT_RATIO_2048, ASPECT_RATIO_2880]:
                self.interpolate_model = T.InterpolationMode.LANCZOS
            self.ratio_index = {}
            self.ratio_nums = {}
            for k, v in self.aspect_ratio.items():
                self.ratio_index[float(k)] = []     # used for self.getitem
                self.ratio_nums[float(k)] = 0      # used for batch-sampler

    def __len__(self):
        return len(self.scene_folders)

    def __getitem__(self, idx):
        try:
            scene_path = self.scene_folders[idx]

            if os.path.exists(os.path.join(scene_path, "images")):
                image_folder = os.path.join(scene_path, "images")
                downscale_factor = 1
            elif os.path.exists(os.path.join(scene_path, "images_4")):
                image_folder = os.path.join(scene_path, "images_4")
                downscale_factor = 1 / 4
            elif os.path.exists(os.path.join(scene_path, "images_8")):
                image_folder = os.path.join(scene_path, "images_8")
                downscale_factor = 1 / 8
            else:
                raise NotImplementedError
            
            json_path = os.path.join(scene_path, "transforms.json")
            caption_path = os.path.join(scene_path, "caption.json")
            image_files = os.listdir(image_folder)
            
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                height, width = json_data['h'], json_data['w']
                
                dh, dw = int(height * downscale_factor), int(width * downscale_factor)
                fl_x, fl_y = json_data['fl_x'] * downscale_factor, json_data['fl_y'] * downscale_factor
                cx = dw // 2
                cy = dh // 2
                
                frame_list = json_data['frames']
            
            # Randomly select number of samples
            
            samples_per_set = random.randint(self.samples_range[0], self.samples_range[1])
            
            # uniformly for all scenes
            if self.frame_sliding_windows is None:
                selected_indices = random.sample(range(len(frame_list)), min(samples_per_set, len(frame_list)))
            # limit the multiview to be in a sliding window (to avoid catastrophic difference in camera angles)
            else:
                # Determine the starting index of the sliding window
                if len(frame_list) <= self.frame_sliding_windows:
                    # If the frame list is smaller than or equal to X, use the entire list
                    window_start = 0
                    window_end = len(frame_list)
                else:
                    # Randomly select a starting point for the window
                    window_start = random.randint(0, len(frame_list) - self.frame_sliding_windows)
                    window_end = window_start + self.frame_sliding_windows

                # Get the indices within the sliding window
                window_indices = list(range(window_start, window_end))

                # Randomly sample indices from the window
                selected_indices = random.sample(window_indices, samples_per_set)
            
            image_files = [os.path.basename(frame_list[i]['file_path']) for i in selected_indices]
            image_paths = [os.path.join(image_folder, file) for file in image_files]
            
            # Load images and convert RGBA to RGB with white background
            images = [convert_rgba_to_rgb_white_bg(Image.open(image_path)) for image_path in image_paths]
            
            if self.transform:
                images = [self.transform(image) for image in images]
            else:
                closest_size, closest_ratio = self.aspect_ratio['1.0'], 1.0
                closest_size = tuple(map(int, closest_size))
                transform = T.Compose([
                            T.ToTensor(),
                            CenterCropResizeImage(closest_size),
                            T.Normalize([.5], [.5]),
                        ])
                images = [transform(image) for image in images]
            images = torch.stack(images)
            
            c2ws = [frame_list[i]['transform_matrix'] for i in selected_indices]
            c2ws = torch.tensor(c2ws).reshape(-1, 4, 4)
            # max_distance = json_data.get('max_distance', self.default_max_distance)
            # if 'max_distance' not in json_data.keys():
                # print(f"not found `max_distance` in json path: {json_path}")

            if self.swap_xz:
                swap_xz = torch.tensor([[[0, 0, 1., 0],
                        [0, 1., 0, 0],
                        [-1., 0, 0, 0],
                        [0, 0, 0, 1.]]])
                c2ws = swap_xz @ c2ws
            
            # OPENGL to OPENCV
            c2ws[:, 0:3, 1:3] *= -1
            c2ws = c2ws[:, [1, 0, 2, 3], :]
            c2ws[:, 2, :] *= -1

            w2cs = torch.inverse(c2ws)
            K = torch.tensor([[[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]]]).repeat(len(c2ws), 1, 1)
            Rs = w2cs[:, :3, :3]
            Ts = w2cs[:, :3, 3]
            sizes = torch.tensor([[dh, dw]]).repeat(len(c2ws), 1)
            
            # get ray embedding and padding last dimension to 16 (num channels of VAE)
            # rays_od = calculate_rays(K, sizes, Rs, Ts, closest_size[0] // 8, use_plucker=False, do_normalize_cameras=self.do_normalize, normalize_scale=self.c2w_scaling)
            rays = calculate_rays(K, sizes, Rs, Ts, closest_size[0] // 8, do_normalize_cameras=self.do_normalize, normalize_scale=self.c2w_scaling)
            rays = rays.reshape(samples_per_set, closest_size[0] // 8, closest_size[1] // 8, 6)
            # padding = (0, 10)  # pad the last dimension to 16
            # rays = torch.nn.functional.pad(rays, padding, "constant", 0)
            rays = torch.cat([rays, rays, rays[..., :4]], dim=-1) * 1.658
            
            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    caption_key = random.choice(self.caption_keys)
                    caption = json.load(f).get(caption_key, "")
            else:
                caption = ""
            
            caption = "[[multiview]] " + caption if caption else "[[multiview]]"
            
            return {
                'pixel_values': images,
                'rays': rays,
                'aspect_ratio': closest_ratio,
                'caption': caption,
                'height': dh,
                'width': dw,
                # 'origins': rays_od[..., :3],
                # 'dirs': rays_od[..., 3:6]
            }
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.scene_folders) - 1))
        