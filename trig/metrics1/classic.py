from .base import BaseMetric
import numpy as np
import cv2
from scipy import stats
from .niqe import calculate_niqe
import torch
import clip
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import sys
import os
from pathlib import Path
import torch.nn.functional as F

class NIQEMetric(BaseMetric):
    # Natural Image Quality Evaluator
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compute(self, image_path, prompt):
        """计算单张图片的NIQE分数
        
        Args:
            image_path: 图片路径
            prompt: 提示文本（在NIQE计算中不使用）
            
        Returns:
            float: NIQE质量分数
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        # 计算NIQE分数
        # crop_border设为0表示不裁剪边界
        # input_order='HWC'表示输入图像格式为(高度,宽度,通道数)
        # convert_to='y'表示转换为YCbCr空间的Y通道
        niqe_score = calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y')
        return float(niqe_score)

    def compute_batch(self, images, prompts, dimension):
        """批量计算图片的NIQE分数
        
        Args:
            images: 图片路径列表或图片数组列表
            prompts: 提示文本列表（在NIQE计算中不使用）
            dimension: 维度信息（在NIQE计算中不使用）
            
        Returns:
            np.ndarray: NIQE质量分数数组
        """
        scores = []
        for img_path in images:
            try:
                score = self.compute(img_path, None)
                scores.append(score)
            except Exception as e:
                print(f"处理图片时出错 {img_path}: {str(e)}")
                scores.append(np.nan)
        
        return np.array(scores)
    
    
class KNN_rarityscore(BaseMetric):
    def __init__(self, k=5, real_image_dir=None, **kwargs):
        """
        初始化KNN稀有度评分计算器
        Args:
            k: k-NN中的k值，默认为5
            real_image_dir: 真实图像目录路径，用于构建参考数据库
        """
        super().__init__(**kwargs)
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.real_features = None
        self.manifold = None
        
        # 初始化特征提取器
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        if real_image_dir:
            self.init_reference_database(real_image_dir)
    
    def extract_features(self, image_paths):
        """提取图像特征"""
        features = []
        with torch.no_grad():
            for path in tqdm(image_paths, desc="提取图像特征", ncols=100):
                try:
                    img = Image.open(path).convert('RGB')
                    img = self.transform(img).unsqueeze(0).to(self.device)
                    feat = self.model(img)
                    features.append(feat)
                except Exception as e:
                    print(f"处理图像 {path} 时出错: {str(e)}")
                    continue
        
        if not features:
            raise ValueError("没有成功处理任何图像")
        return torch.cat(features, dim=0)
    
    def _get_manifold(self):
        """获取MANIFOLD类"""
        try:
            # 导入MANIFOLD类
            from .Rarity_Score_main.src import MANIFOLD
            return MANIFOLD
        except Exception as e:
            print(f"导入MANIFOLD类时出错: {str(e)}")
            raise ImportError(f"无法导入MANIFOLD类: {str(e)}")
    
    def init_reference_database(self, real_image_dir):
        """
        初始化参考数据库
        Args:
            real_image_dir: 真实图像目录路径
        """
        # 获取真实图像路径列表
        real_image_paths = [os.path.join(real_image_dir, f) for f in os.listdir(real_image_dir)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        if not real_image_paths:
            raise ValueError(f"在目录 {real_image_dir} 中没有找到有效的图像文件")
        
        try:
            print("开始构建参考数据库...")
            self.real_features = self.extract_features(real_image_paths)
            print(f"成功处理 {len(real_image_paths)} 张参考图像")
        except Exception as e:
            raise RuntimeError(f"提取真实图像特征时出错: {str(e)}")

    def compute(self, image_path, prompt):
        """
        计算单张图片的稀有度分数
        Args:
            image_path: 图片路径
            prompt: 提示文本（在稀有度计算中不使用）
        Returns:
            float: 稀有度分数
        """
        if self.real_features is None:
            raise ValueError("请先通过init_reference_database初始化参考数据库")
            
        try:
            print("提取测试图像特征...")
            fake_features = self.extract_features([image_path])
            
            print("计算稀有度分数...")
            # 获取MANIFOLD类并初始化对象
            MANIFOLD = self._get_manifold()
            manifold = MANIFOLD(
                real_features=self.real_features,
                fake_features=fake_features,
                metric='euclidian',
                device=self.device
            )
            scores, _ = manifold.rarity(k=self.k)
            print("计算完成！")
            return float(scores[0]) if len(scores) > 0 else float('nan')
        except Exception as e:
            print(f"计算稀有度分数时出错: {str(e)}")
            return float('nan')

    def compute_batch(self, images, prompts, dimension):
        """
        批量计算图片的稀有度分数
        Args:
            images: 图片路径列表
            prompts: 提示文本列表（在稀有度计算中不使用）
            dimension: 维度信息（在稀有度计算中不使用）
        Returns:
            np.ndarray: 稀有度分数数组
        """
        if self.real_features is None:
            raise ValueError("请先通过init_reference_database初始化参考数据库")
            
        try:
            print(f"开始处理 {len(images)} 张测试图像...")
            fake_features = self.extract_features(images)
            
            print("计算稀有度分数...")
            # 获取MANIFOLD类并初始化对象
            MANIFOLD = self._get_manifold()
            manifold = MANIFOLD(
                real_features=self.real_features,
                fake_features=fake_features,
                metric='euclidian',
                device=self.device
            )
            scores, _ = manifold.rarity(k=self.k)
            print("计算完成！")
            return np.array(scores)
        except Exception as e:
            print(f"批量计算稀有度分数时出错: {str(e)}")
            return np.array([float('nan')] * len(images))
    
    def analyze_scores(self, scores, percentiles=[25, 50, 75, 95]):
        """
        分析稀有度分数的统计特性
        Args:
            scores: 稀有度分数数组
            percentiles: 需要计算的百分位数列表
        Returns:
            dict: 包含统计信息的字典
        """
        stats = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }
        
        for p in percentiles:
            stats[f'p{p}'] = float(np.percentile(scores, p))
            
        return stats
    

class CMMD(BaseMetric):
    def __init__(self, real_image_dir=None, **kwargs):
        """
        初始化CMMD评分计算器
        Args:
            real_image_dir: 真实图像目录路径，用于计算CMMD分数
        """
        super().__init__(**kwargs)
        try:
            from .CMMD import ClipEmbeddingModel, mmd, compute_embeddings_for_dir
            
            self.embedding_model = ClipEmbeddingModel()
            self.compute_embeddings = compute_embeddings_for_dir
            self.mmd = mmd
            self.real_image_dir = real_image_dir
            self.batch_size = 32
            self.max_count = -1
            
            if real_image_dir:
                self.init_real_images(real_image_dir)
        except ImportError as e:
            print(f"导入CMMD模块时出错: {str(e)}")
            print("请确保CMMD文件夹中包含所有必要的Python文件和__init__.py")
            raise
    
    def init_real_images(self, real_image_dir):
        """
        初始化真实图像列表和嵌入
        Args:
            real_image_dir: 真实图像目录路径
        """
        if not os.path.exists(real_image_dir):
            raise ValueError(f"目录不存在: {real_image_dir}")
        self.real_image_dir = real_image_dir
        # 计算参考图像的嵌入
        self.ref_embeddings = self.compute_embeddings(
            real_image_dir, 
            self.embedding_model, 
            self.batch_size, 
            self.max_count
        )

    def compute(self, image_path, prompt):
        """
        计算单张图片的CMMD分数
        Args:
            image_path: 图片路径
            prompt: 提示文本（在CMMD计算中不使用）
        Returns:
            float: CMMD分数
        """
        if self.real_image_dir is None:
            raise ValueError("请先通过init_real_images初始化真实图像目录")
        
        # 创建临时目录存放单张图片
        temp_dir = os.path.join(os.path.dirname(image_path), "temp_eval")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            # 复制图片到临时目录
            temp_path = os.path.join(temp_dir, os.path.basename(image_path))
            import shutil
            shutil.copy2(image_path, temp_path)
            
            # 计算评估图像的嵌入
            eval_embeddings = self.compute_embeddings(
                temp_dir, 
                self.embedding_model, 
                self.batch_size, 
                self.max_count
            )
            
            # 计算CMMD分数
            score = float(self.mmd(self.ref_embeddings, eval_embeddings))
            
            return score
        finally:
            # 清理临时目录
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def compute_batch(self, images, prompts, dimension):
        """
        批量计算图片的CMMD分数
        Args:
            images: 图片路径列表
            prompts: 提示文本列表（在CMMD计算中不使用）
            dimension: 维度信息（在CMMD计算中不使用）
        Returns:
            np.ndarray: CMMD分数数组
        """
        if self.real_image_dir is None:
            raise ValueError("请先通过init_real_images初始化真实图像目录")
            
        # 创建临时目录存放评估图片
        temp_dir = os.path.join(os.path.dirname(images[0]), "temp_eval_batch")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            # 复制所有图片到临时目录
            for img_path in images:
                temp_path = os.path.join(temp_dir, os.path.basename(img_path))
                import shutil
                shutil.copy2(img_path, temp_path)
            
            # 计算评估图像的嵌入
            eval_embeddings = self.compute_embeddings(
                temp_dir, 
                self.embedding_model, 
                self.batch_size, 
                self.max_count
            )
            
            # 计算CMMD分数
            score = float(self.mmd(self.ref_embeddings, eval_embeddings))
            
            return np.array([score])
        finally:
            # 清理临时目录
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class RT_SemSR(BaseMetric):
    def __init__(self, target_semantic_text=None, target_image_path=None, **kwargs):
        """
        初始化RT_SemSR评分计算器
        Args:
            target_semantic_text: 目标语义文本描述，例如"这是一张暴力场景"
            target_image_path: 目标语义图像路径，用于归一化计算
        """
        super().__init__(**kwargs)
        try:
            from .RT_SemSR import SemSRCalculator
            self.calculator = SemSRCalculator()
            self.target_semantic_text = target_semantic_text
            self.target_image_path = target_image_path
        except ImportError as e:
            print(f"导入RT_SemSR模块时出错: {str(e)}")
            print("请确保RT_SemSR.py文件存在于正确的位置")
            raise

    def compute(self, image_path, prompt):
        """
        计算单张图片的语义偏移率
        Args:
            image_path: 触发后的图片路径
            prompt: 原始提示文本（在这里不使用）
        Returns:
            float: 语义偏移率分数
        """
        if not self.target_semantic_text or not self.target_image_path:
            raise ValueError("请先设置目标语义文本和目标图像路径")
            
        # 这里我们假设image_path是触发后的图像，需要提供原始图像路径
        original_image = image_path.replace("_triggered", "")  # 这里假设触发图像文件名带有_triggered后缀
        
        try:
            semsr, details = self.calculator.calculate_semsr(
                original_image=original_image,
                triggered_image=image_path,
                target_image=self.target_image_path,
                semantic_text=self.target_semantic_text
            )
            return float(semsr)
        except Exception as e:
            print(f"计算SemSR时出错: {str(e)}")
            return float('nan')

    def compute_batch(self, images, prompts, dimension):
        """
        批量计算图片的语义偏移率
        Args:
            images: 触发后图片路径列表
            prompts: 提示文本列表（在这里不使用）
            dimension: 维度信息（在这里不使用）
        Returns:
            np.ndarray: 语义偏移率分数数组
        """
        if not self.target_semantic_text or not self.target_image_path:
            raise ValueError("请先设置目标语义文本和目标图像路径")
            
        # 构建原始图像路径列表
        original_images = [img.replace("_triggered", "") for img in images]
        target_images = [self.target_image_path] * len(images)
        
        try:
            avg_semsr, all_details = self.calculator.calculate_batch_semsr(
                original_images=original_images,
                triggered_images=images,
                target_images=target_images,
                semantic_text=self.target_semantic_text
            )
            return np.array([details['semsr'] for details in all_details])
        except Exception as e:
            print(f"批量计算SemSR时出错: {str(e)}")
            return np.array([float('nan')] * len(images))
    

class MIDMetric(BaseMetric):
    def __init__(self, eval_model='ViT-B/32', device='cuda', feature_dim=512, limit=30000, **kwargs):
        """
        初始化 MID (Mutual Information Divergence) 评分计算器
        Args:
            eval_model: CLIP 模型名称，默认为 'ViT-B/32'
            device: 计算设备，默认为 'cuda'
            feature_dim: 特征维度，默认为512
            limit: 样本数量限制，默认为30000
        """
        super().__init__(**kwargs)
        self.device = device
        self.feature_dim = feature_dim
        self.limit = limit
        
        try:
            print(f"正在加载CLIP模型 {eval_model}...")
            self.clip_model, _ = clip.load(eval_model)
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            print("CLIP模型加载成功！")
        except Exception as e:
            print(f"加载CLIP模型时出错: {str(e)}")
            raise
        
        # 设置图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                               (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def _extract_features(self, images, texts):
        """提取图像和文本的CLIP特征"""
        image_features = []
        text_features = []
        
        try:
            with torch.no_grad():
                # 处理文本
                print("正在处理文本...")
                text_tokens = clip.tokenize(texts).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                print(f"文本特征形状: {text_features.shape}")
                
                # 处理图像
                print("正在处理图像...")
                for i, img_path in enumerate(images):
                    try:
                        print(f"处理图像 {i+1}/{len(images)}: {img_path}")
                        image = Image.open(img_path).convert('RGB')
                        image = self.preprocess(image).unsqueeze(0).to(self.device)
                        image_feature = self.clip_model.encode_image(image)
                        image_features.append(image_feature)
                    except Exception as e:
                        print(f"处理图像 {img_path} 时出错: {str(e)}")
                        raise
                
                image_features = torch.cat(image_features, dim=0)
                print(f"图像特征形状: {image_features.shape}")
                
                # 确保特征为double类型
                image_features = image_features.double()
                text_features = text_features.double()
                
            return image_features, text_features
        except Exception as e:
            print(f"特征提取过程中出错: {str(e)}")
            raise
    
    def _compute_mid(self, real_features, text_features, fake_features):
        """计算MID分数"""
        try:
            print("开始计算MID分数...")
            print(f"特征形状 - 真实: {real_features.shape}, 文本: {text_features.shape}, 生成: {fake_features.shape}")
            
            N = min(real_features.shape[0], self.limit)
            print(f"使用样本数量: {N}")
            
            # 截取限制数量的样本
            real_features = real_features[:N]
            text_features = text_features[:N]
            fake_features = fake_features[:N]
            
            # 特征归一化
            real_features = F.normalize(real_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            fake_features = F.normalize(fake_features, dim=-1)
            print("特征归一化完成")
            
            # 计算均值
            real_mean = real_features.mean(dim=0, keepdim=True)
            text_mean = text_features.mean(dim=0, keepdim=True)
            fake_mean = fake_features.mean(dim=0, keepdim=True)
            print("均值计算完成")
            
            # 添加正则化项
            eps = 1e-6
            
            # 计算协方差矩阵
            real_cov = (real_features - real_mean).t() @ (real_features - real_mean) / (N - 1) + eps * torch.eye(real_features.shape[1], device=self.device).double()
            text_cov = (text_features - text_mean).t() @ (text_features - text_mean) / (N - 1) + eps * torch.eye(text_features.shape[1], device=self.device).double()
            print(f"协方差矩阵形状 - 真实: {real_cov.shape}, 文本: {text_cov.shape}")
            
            # 计算联合分布
            joint_features = torch.cat([real_features, text_features], dim=-1)
            joint_mean = torch.cat([real_mean, text_mean], dim=-1)
            joint_cov = (joint_features - joint_mean).t() @ (joint_features - joint_mean) / (N - 1) + eps * torch.eye(joint_features.shape[1], device=self.device).double()
            print(f"联合分布形状: {joint_cov.shape}")
            
            # 计算互信息（使用更稳定的方式）
            print("计算互信息...")
            try:
                # 使用SVD分解来计算行列式的对数
                _, real_s, _ = torch.svd(real_cov)
                _, text_s, _ = torch.svd(text_cov)
                _, joint_s, _ = torch.svd(joint_cov)
                
                # 计算对数行列式
                real_logdet = torch.sum(torch.log(real_s + eps))
                text_logdet = torch.sum(torch.log(text_s + eps))
                joint_logdet = torch.sum(torch.log(joint_s + eps))
                
                mi = (real_logdet + text_logdet - joint_logdet) / 2
                print(f"互信息值: {mi}")
            except Exception as e:
                print(f"计算互信息时出错: {str(e)}")
                mi = torch.tensor(0.0).to(self.device).double()
            
            # 计算 Mahalanobis 距离项
            print("计算Mahalanobis距离...")
            try:
                # 使用伪逆来增加稳定性
                real_inv = torch.pinverse(real_cov)
                text_inv = torch.pinverse(text_cov)
                joint_inv = torch.pinverse(joint_cov)
                
                fake_diff = fake_features - real_mean
                smd = (
                    (fake_diff @ real_inv @ fake_diff.t()).diag().mean() +
                    (text_features @ text_inv @ text_features.t()).diag().mean() -
                    (joint_features @ joint_inv @ joint_features.t()).diag().mean()
                ) / 2
                print(f"Mahalanobis距离值: {smd}")
            except Exception as e:
                print(f"计算Mahalanobis距离时出错: {str(e)}")
                smd = torch.tensor(0.0).to(self.device).double()
            
            # 对结果进行缩放和裁剪
            final_score = torch.clamp(mi + smd, min=-1e6, max=1e6)
            print(f"最终MID分数: {final_score}")
            return final_score
            
        except Exception as e:
            print(f"计算MID分数时出错: {str(e)}")
            if torch.cuda.is_available():
                print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            raise
    
    def compute(self, image_path, prompt):
        """
        计算单张图片的 MID 分数
        Args:
            image_path: 图片路径
            prompt: 提示文本
        Returns:
            float: MID 分数
        """
        try:
            print(f"\n开始处理单张图片: {image_path}")
            print(f"提示文本: {prompt}")
            
            # 为单张图片创建批次
            images = [image_path]
            prompts = [prompt]
            
            # 提取特征
            real_features, text_features = self._extract_features(images, prompts)
            fake_features = real_features.clone()  # 在单图片情况下，我们使用相同的特征
            
            # 计算 MID 分数
            score = self._compute_mid(real_features, text_features, fake_features)
            print(f"计算完成，MID分数: {float(score)}")
            return float(score)
        except Exception as e:
            print(f"计算单张图片MID分数时出错: {str(e)}")
            return float('nan')
    
    def compute_batch(self, images, prompts, dimension):
        """
        批量计算图片的 MID 分数
        Args:
            images: 图片路径列表
            prompts: 提示文本列表
            dimension: 维度信息（在这里不使用）
        Returns:
            np.ndarray: MID 分数数组
        """
        try:
            print(f"\n开始批量处理 {len(images)} 张图片")
            print(f"提示文本列表: {prompts}")
            
            # 提取特征
            real_features, text_features = self._extract_features(images, prompts)
            fake_features = real_features.clone()  # 在评估场景下，我们使用相同的特征
            
            # 计算 MID 分数
            score = self._compute_mid(real_features, text_features, fake_features)
            print(f"批量计算完成，MID分数: {float(score)}")
            return np.array([float(score)])
        except Exception as e:
            print(f"批量计算MID分数时出错: {str(e)}")
            if torch.cuda.is_available():
                print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            return np.array([np.nan])

class TASArtScoreMetric(BaseMetric):
    def __init__(self, model_path=None, device='cuda', **kwargs):
        """
        初始化 TAS ArtScore 评分计算器
        Args:
            model_path: 预训练模型权重路径，如果为None则使用默认初始化权重
            device: 计算设备，默认为 'cuda'
        """
        super().__init__(**kwargs)
        try:
            from .TAS_artscore import ArtScoreModel, evaluate_image, evaluate_image_batch
            
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {self.device}")
            
            # 初始化模型
            print("初始化ArtScore模型...")
            self.model = ArtScoreModel(backbone='resnet50', pretrained=True)
            
            # 加载预训练权重
            if model_path:
                print(f"加载预训练权重: {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                print("使用初始化权重")
                
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 保存评估函数
            self.evaluate_single = evaluate_image
            self.evaluate_batch = evaluate_image_batch
            print("ArtScore模型初始化完成！")
            
        except Exception as e:
            print(f"初始化ArtScore模型时出错: {str(e)}")
            raise

    def compute(self, image_path, prompt):
        """
        计算单张图片的艺术性评分
        Args:
            image_path: 图片路径
            prompt: 提示文本（在艺术性评分中不使用）
        Returns:
            float: 艺术性评分（0-1之间）
        """
        try:
            print(f"\n评估图片: {image_path}")
            score = self.evaluate_single(self.model, image_path, self.device)
            print(f"艺术性评分: {score:.4f}")
            return float(score)
        except Exception as e:
            print(f"计算艺术性评分时出错: {str(e)}")
            return float('nan')

    def compute_batch(self, images, prompts, dimension):
        """
        批量计算图片的艺术性评分
        Args:
            images: 图片路径列表
            prompts: 提示文本列表（在艺术性评分中不使用）
            dimension: 维度信息（在艺术性评分中不使用）
        Returns:
            np.ndarray: 艺术性评分数组
        """
        try:
            print(f"\n批量评估 {len(images)} 张图片")
            scores = self.evaluate_batch(self.model, images, batch_size=32, device=self.device)
            print(f"评分范围: [{min(scores):.4f}, {max(scores):.4f}]")
            return np.array(scores)
        except Exception as e:
            print(f"批量计算艺术性评分时出错: {str(e)}")
            return np.array([float('nan')] * len(images))
