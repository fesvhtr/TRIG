import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Union
import os
import torchvision.transforms as T

class CMMMDCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', sigma=10.0):
        """
        初始化CMMD计算器
        Args:
            device: 计算设备
            sigma: RBF核的带宽参数
        """
        self.device = device
        self.sigma = sigma
        self.gamma = 1.0 / (2 * sigma**2)
        # 加载CLIP模型，使用与原始实现相同的模型
        self.model, _ = clip.load("ViT-L/14@336px", device=self.device)
        # 自定义预处理步骤，模仿原始实现
        self.preprocess = T.Compose([
            T.Resize(336, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(336),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                       (0.26862954, 0.26130258, 0.27577711))
        ])
        
    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        提取图像的CLIP特征
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小
        Returns:
            torch.Tensor: 特征向量张量
        """
        features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = self.preprocess(image).unsqueeze(0)
                    batch_images.append(image)
                except Exception as e:
                    print(f"处理图片时出错 {img_path}: {str(e)}")
                    continue
            
            if not batch_images:
                continue
                
            batch_images = torch.cat(batch_images, dim=0).to(self.device)
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_images)
            features.append(batch_features)
        
        if not features:
            raise ValueError("没有成功处理任何图片")
            
        features = torch.cat(features, dim=0)
        return features.float()

    def calculate_mmd(self, real_features: torch.Tensor, gen_features: torch.Tensor) -> float:
        """
        计算MMD距离，使用更高效的实现
        Args:
            real_features: 真实图像特征
            gen_features: 生成图像特征
        Returns:
            float: MMD距离
        """
        # 计算平方范数
        x_sqnorms = (real_features**2).sum(dim=1)
        y_sqnorms = (gen_features**2).sum(dim=1)
        
        # 计算核矩阵，使用分块计算以节省内存
        k_xx = torch.exp(-self.gamma * (
            -2 * torch.mm(real_features, real_features.t()) +
            x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0)
        )).mean()
        
        k_yy = torch.exp(-self.gamma * (
            -2 * torch.mm(gen_features, gen_features.t()) +
            y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0)
        )).mean()
        
        k_xy = torch.exp(-self.gamma * (
            -2 * torch.mm(real_features, gen_features.t()) +
            x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0)
        )).mean()
        
        # 计算MMD，直接使用1000作为缩放因子
        mmd = 1000 * (k_xx + k_yy - 2 * k_xy)
        return mmd.item()

    def calculate_cmmd(self, real_images: List[str], gen_images: List[str], 
                      batch_size: int = 32) -> float:
        """
        计算CMMD分数
        Args:
            real_images: 真实图像路径列表
            gen_images: 生成图像路径列表
            batch_size: 批处理大小
        Returns:
            float: CMMD分数
        """
        # 提取特征
        real_features = self.extract_features(real_images, batch_size)
        gen_features = self.extract_features(gen_images, batch_size)
        
        # 计算MMD距离
        cmmd_score = self.calculate_mmd(real_features, gen_features)
        return cmmd_score
