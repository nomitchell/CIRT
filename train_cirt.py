import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchattacks import PGD
from tqdm import tqdm
from models.cirt_model import CIRT_Model
import argparse

# --- 1. Contrastive Invariance Loss (NT-Xent) ---
class InvarianceLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        # z1: projections of clean batch, z2: projections of style-augmented batch
        batch_size = z1.shape[0]
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # Concatenate features for contrastive learning
        features = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels: positives are (z1[i], z2[i]) pairs
        # The original labels point to the positive pairs in the original similarity matrix
        labels = torch.arange(batch_size).to(z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask to remove self-similarity from loss calculation
        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z1.device)
        similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)
        
        # Adjust the labels to account for the removed diagonal elements
        # For the first half of the batch (z1), the positive key (z2) is at index batch_size - 1
        # For the second half of the batch (z2), the positive key (z1) is at index batch_size
        # After removing the diagonal, these shift.
        # The positive key for z1[i] was at index batch_size + i. After removing the main diagonal,
        # the first batch_size elements are removed, so the index becomes i.
        # The positive key for z2[i] was at index i. After removing the main diagonal,
        # the first batch_size elements are removed, so the index becomes i.
        # The labels should now be the same for both halves of the batch.
        labels = torch.arange(batch_size).to(z1.device)

        loss = self.criterion(similarity_matrix, torch.cat([labels, labels]))
        return loss

# --- Wrapper for TorchAttacks ---
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits

# --- 2. Main Training Loop ---
def train_cirt(epochs, device, trainloader, model, optimizer, attack, style_transform, lambda_invariance, lambda_noise):
    # Define Loss Functions
    criterion_adv = nn.CrossEntropyLoss()
    criterion_invariance = InvarianceLoss()
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # --- Generate the three required inputs for this batch ---
            # 1. Adversarial input (for L_Adversarial)
            adv_inputs = attack(inputs, targets)
            # 2. Style-augmented input (for L_Invariance)
            style_inputs = style_transform(inputs)
            # 3. Noisy input (for L_Noise)
            noisy_inputs = (inputs + torch.randn_like(inputs) * 0.1).clamp(0, 1)

            optimizer.zero_grad()
            
            # --- Forward passes ---
            logits_adv, _ = model(adv_inputs)
            _, projection_clean = model(inputs)
            _, projection_style = model(style_inputs)
            logits_noisy, _ = model(noisy_inputs)

            # --- Calculate the three-part loss ---
            loss_adv = criterion_adv(logits_adv, targets)
            loss_invariance = criterion_invariance(projection_clean, projection_style)
            loss_noise = criterion_adv(logits_noisy, targets)
            
            # --- Combine losses (lambdas are key hyperparameters) ---
            loss = loss_adv + lambda_invariance * loss_invariance + lambda_noise * loss_noise
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), 'cirt_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIRT Training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--lambda-invariance', type=float, default=0.5, help='lambda for invariance loss')
    parser.add_argument('--lambda-noise', type=float, default=0.5, help='lambda for noise loss')
    parser.add_argument('--data-dir', type=str, default='./data', help='directory for dataset')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Model
    model = CIRT_Model().to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Adversarial Attack
    attack_model = ModelWrapper(model)
    attack = PGD(attack_model, eps=8/255, alpha=2/255, steps=10)

    # Style Transform
    style_transform = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.4)

    train_cirt(args.epochs, device, trainloader, model, optimizer, attack, style_transform, args.lambda_invariance, args.lambda_noise)

    # Save model
    torch.save(model.state_dict(), 'cirt_model.pth')
