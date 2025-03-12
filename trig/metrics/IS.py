import torch
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

class ImageDataset(Dataset):
    """Dataset for loading images containing 'realism' in the filename."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if 'realism' in f.lower() and f.endswith(('png', 'jpg', 'jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def calculate_inception_score(image_dir, batch_size=32, splits=10):
    """Computes Inception Score (IS) for a folder of images containing 'realism' in the filename."""
    # Load pre-trained InceptionV3 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load images
    dataset = ImageDataset(image_dir, transform=transform)
    
    if len(dataset) == 0:
        print("No images containing 'realism' found in the dataset.")
        return None, None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    preds = []
    
    # Compute predictions
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = inception_model(batch)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())  # Convert logits to probabilities
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute Inception Score
    split_scores = []
    for k in range(splits):
        part = preds[k * (preds.shape[0] // splits): (k + 1) * (preds.shape[0] // splits), :]
        p_y = np.mean(part, axis=0, keepdims=True)  # Compute mean distribution
        kl_div = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))  # KL divergence
        split_scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))  # Compute IS for each split
    
    # Compute mean and standard deviation of IS
    is_mean, is_std = np.mean(split_scores), np.std(split_scores)
    return is_mean, is_std

# Example usage
image_directory = "/home/muzammal/Projects/TRIG/data/output/HEIM/sd35"  # Change this to your image folder path
is_mean, is_std = calculate_inception_score(image_directory)
if is_mean is not None:
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
