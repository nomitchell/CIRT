import logging
from utils import setup_logger
import optuna
from train_cirt import train_cirt, InvarianceLoss
from models.cirt_model import CIRT_Model
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.nn as nn
from torchvision.datasets import DTD
from torchattacks import PGD

def evaluate(model, loader, attack=None):
    correct = 0
    total = 0
    model.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if attack:
            inputs = attack(inputs, targets)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100. * correct / total

def objective(trial):
    lambda_invariance = trial.suggest_float('lambda_invariance', 1e-2, 1.0, log=True)
    lambda_noise = trial.suggest_float('lambda_noise', 1e-2, 1.0, log=True)
    
    model = CIRT_Model().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Use a shorter training schedule for HPO
    train_cirt(model, trainloader, texture_loader, epochs=20, lr=1e-3,
               lambda_invariance=lambda_invariance, lambda_noise=lambda_noise, device=device)
    
    # Evaluate robust accuracy on a validation set
    attack = PGD(model, eps=8/255, alpha=2/255, steps=7)
    robust_acc = evaluate(model, valloader, attack=attack)
    trial.set_user_attr("robust_acc", robust_acc)
    logging.info(f"Trial {trial.number} finished with robust accuracy: {robust_acc:.2f}% and params: {trial.params}")
    return robust_acc

if __name__ == '__main__':
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_set, val_set = random_split(train_set, [45000, 5000])
    trainloader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

    texture_transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    texture_dataset = DTD(root='./data', split='train', download=True, transform=texture_transform)
    texture_dataset_full = ConcatDataset([texture_dataset] * (len(train_set) // len(texture_dataset) + 1))
    texture_loader = DataLoader(texture_dataset_full, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    logging.info(f"Best HPO params: {study.best_params}")
