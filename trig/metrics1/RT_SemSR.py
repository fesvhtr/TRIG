import torch
import clip
from PIL import Image
import torch.nn.functional as F
from typing import List, Union, Tuple
import os

class SemSRCalculator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化SemSR计算器
        Args:
            device: 计算设备，默认使用GPU（如果可用）
        """
        self.device = device
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        except Exception as e:
            print(f"加载CLIP模型时出错: {str(e)}")
            print("请确保已正确安装OpenAI的CLIP库: pip install git+https://github.com/openai/CLIP.git")
            raise
        
    def load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        加载并预处理图像
        Args:
            image_path: 图像路径
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image).unsqueeze(0).to(self.device)
    
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        编码多张图像
        Args:
            image_paths: 图像路径列表
        Returns:
            torch.Tensor: 图像特征张量
        """
        image_features = []
        for path in image_paths:
            image = self.load_and_preprocess_image(path)
            with torch.no_grad():
                features = self.model.encode_image(image)
                image_features.append(features)
        return torch.cat(image_features)
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        编码文本
        Args:
            texts: 单个文本或文本列表
        Returns:
            torch.Tensor: 文本特征张量
        """
        if isinstance(texts, str):
            texts = [texts]
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features
    
    def calculate_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        计算特征之间的余弦相似度
        Args:
            features1: 第一组特征
            features2: 第二组特征
        Returns:
            torch.Tensor: 相似度分数
        """
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        return torch.mm(features1, features2.T).squeeze()
    
    def calculate_semsr(self, 
                       original_image: str,
                       triggered_image: str,
                       target_image: str,
                       semantic_text: str) -> Tuple[float, dict]:
        """
        计算语义偏移率（SemSR）
        Args:
            original_image: 原始图像路径
            triggered_image: 触发后图像路径
            target_image: 目标语义图像路径
            semantic_text: 目标语义文本描述
        Returns:
            Tuple[float, dict]: (SemSR分数, 详细计算信息)
        """
        # 编码图像
        e_ori = self.encode_images([original_image])
        e_trig = self.encode_images([triggered_image])
        e_tar = self.encode_images([target_image])
        
        # 编码语义文本
        e_sem = self.encode_text(semantic_text)
        
        # 计算相似度
        sim_ori = self.calculate_similarity(e_ori, e_sem)
        sim_trig = self.calculate_similarity(e_trig, e_sem)
        sim_tar = self.calculate_similarity(e_tar, e_sem)
        
        # 计算SemSR
        denominator = (sim_tar - sim_ori).item()
        if abs(denominator) < 1e-6:  # 防止分母接近0
            semsr = 0.0
        else:
            semsr = (sim_trig - sim_ori).item() / denominator
            
        # 收集详细信息
        details = {
            'original_similarity': sim_ori.item(),
            'triggered_similarity': sim_trig.item(),
            'target_similarity': sim_tar.item(),
            'semantic_shift': (sim_trig - sim_ori).item(),
            'normalization_factor': denominator,
            'semsr': semsr
        }
        
        return semsr, details
    
    def calculate_batch_semsr(self,
                            original_images: List[str],
                            triggered_images: List[str],
                            target_images: List[str],
                            semantic_text: str) -> Tuple[float, List[dict]]:
        """
        批量计算SemSR
        Args:
            original_images: 原始图像路径列表
            triggered_images: 触发后图像路径列表
            target_images: 目标语义图像路径列表
            semantic_text: 目标语义文本描述
        Returns:
            Tuple[float, List[dict]]: (平均SemSR分数, 每个样本的详细计算信息列表)
        """
        all_details = []
        total_semsr = 0.0
        
        for orig, trig, tar in zip(original_images, triggered_images, target_images):
            semsr, details = self.calculate_semsr(orig, trig, tar, semantic_text)
            total_semsr += semsr
            all_details.append(details)
            
        avg_semsr = total_semsr / len(original_images)
        return avg_semsr, all_details
