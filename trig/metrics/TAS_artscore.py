import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ArtScoreModel(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        """
        ArtScore 模型初始化
        Args:
            backbone: 使用的骨干网络，默认为 ResNet50
            pretrained: 是否使用预训练权重
        """
        super(ArtScoreModel, self).__init__()
        
        # 加载骨干网络
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # 移除最后的全连接层
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.fc_dim = 2048
        else:
            raise NotImplementedError(f"未实现的骨干网络: {backbone}")
        
        # 添加评分头
        self.score_head = nn.Sequential(
            nn.Linear(self.fc_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出范围压缩到 [0,1]
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 tensor, shape [B, C, H, W]
        Returns:
            艺术性评分，范围 [0,1]
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        score = self.score_head(features)
        return score

class ArtScoreDataset(Dataset):
    """用于训练 ArtScore 模型的数据集"""
    def __init__(self, image_paths, scores, transform=None):
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        return image, score

def train_artscore_model(model, train_loader, val_loader, 
                        epochs=100, lr=0.0001, device='cuda'):
    """
    训练 ArtScore 模型
    Args:
        model: ArtScore 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for images, scores in train_loader:
            images, scores = images.to(device), scores.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, scores in val_loader:
                images, scores = images.to(device), scores.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, scores).item()
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_artscore_model.pth')

def evaluate_image(model, image_path, device='cuda'):
    """
    评估单张图像的艺术性得分
    Args:
        model: 训练好的 ArtScore 模型
        image_path: 图像路径
        device: 运行设备
    Returns:
        艺术性得分 (0-1 之间)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        score = model(image)
    
    return score.item()

def evaluate_image_batch(model, image_paths, batch_size=32, device='cuda'):
    """
    批量评估图像的艺术性得分
    Args:
        model: 训练好的 ArtScore 模型
        image_paths: 图像路径列表
        batch_size: 批处理大小
        device: 运行设备
    Returns:
        艺术性得分列表
    """
    dataset = ArtScoreDataset(image_paths, [0]*len(image_paths))  # 虚拟分数
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    scores = []
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            batch_scores = model(images)
            scores.extend(batch_scores.cpu().numpy())
    
    return scores

if __name__ == '__main__':
    # 示例用法
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 创建模型
    model = ArtScoreModel(backbone='resnet50', pretrained=True)
    
    # 2. 加载预训练权重（如果有）
    try:
        model.load_state_dict(torch.load('best_artscore_model.pth'))
        print("已加载预训练模型权重")
    except:
        print("未找到预训练模型权重，使用初始化权重")
    
    # 3. 评估示例
    # image_path = "example.jpg"
    # score = evaluate_image(model, image_path, device)
    # print(f"图像的艺术性得分: {score:.4f}")
