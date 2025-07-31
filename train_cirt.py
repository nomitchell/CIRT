import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import DTD # Describable Textures Dataset
from tqdm import tqdm
from torchattacks import PGD
from torch.cuda.amp import GradScaler, autocast

from models.cirt_model import CIRT_Model

# --- 1. AdaIN Style Augmentation & Contrastive Loss ---
# AdaIN implementation (standard, based on the original paper)
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# NT-Xent Loss for Invariance
class InvarianceLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        features = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create a mask to remove the diagonal elements (self-similarity)
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # The labels are the indices of the positive pairs
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        loss = self.criterion(similarity_matrix, labels)
        return loss

# --- 2. Main Training Function ---
def train_cirt(model, trainloader, texture_loader, epochs, lr, lambda_invariance, lambda_noise, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_invariance = InvarianceLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    attack = PGD(model, eps=8/255, alpha=2/255, steps=10)

    for epoch in range(epochs):
        model.train()
        texture_iter = iter(texture_loader)
        pbar = tqdm(trainloader, desc=f"CIRT Training Epoch {epoch+1}/{epochs}")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            adv_inputs = attack(inputs, targets)
            noisy_inputs = (inputs + torch.randn_like(inputs) * 0.1).clamp(0, 1)
            try:
                style_images, _ = next(texture_iter)
            except StopIteration:
                texture_iter = iter(texture_loader)
                style_images, _ = next(texture_iter)
            style_inputs = adain(inputs, style_images.to(device))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits_adv = model(adv_inputs)
                logits_noisy = model(noisy_inputs)
                _, proj_clean = model(inputs, with_projection=True)
                _, proj_style = model(style_inputs, with_projection=True)

                loss_adv = criterion_ce(logits_adv, targets)
                loss_noise = criterion_ce(logits_noisy, targets)
                loss_invariance = criterion_invariance(proj_clean, proj_style)
                
                loss = loss_adv + lambda_invariance * loss_invariance + lambda_noise * loss_noise
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    texture_transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    texture_dataset = DTD(root='./data', split='train', download=True, transform=texture_transform)
    texture_dataset_full = ConcatDataset([texture_dataset] * (len(train_set) // len(texture_dataset) + 1))
    texture_loader = DataLoader(texture_dataset_full, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    
    cirt_model = CIRT_Model().to(device)
    
    # These lambdas should be determined via HPO
    best_lambda_invariance = 0.1 
    best_lambda_noise = 0.2
    
    train_cirt(cirt_model, trainloader, texture_loader, epochs=100, lr=1e-3, 
               lambda_invariance=best_lambda_invariance, lambda_noise=best_lambda_noise, device=device)
    
    torch.save(cirt_model.state_dict(), "cirt_final.pth")