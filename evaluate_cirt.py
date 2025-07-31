import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from robustbench.utils import load_model
from autoattack import AutoAttack
from models.cirt_model import CIRT_Model
import argparse

def evaluate(model, loader, attack=None):
    correct = 0
    total = 0
    model.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if attack:
            inputs = attack(inputs, targets)
        with torch.no_grad():
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100. * correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIRT Evaluation')
    parser.add_argument('--data-dir', type=str, default='./data', help='directory for dataset')
    parser.add_argument('--model-path', type=str, default='cirt_model.pth', help='path to trained CIRT model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Load CIRT model
    cirt_model = CIRT_Model().to(device)
    cirt_model.load_state_dict(torch.load(args.model_path))
    cirt_model.eval()

    # --- 1. The Leaderboard Test (Preprocessor-Blind) ---
    print("--- Running Preprocessor-Blind Evaluation ---")
    standard_model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf').to(device)
    blind_attack = AutoAttack(standard_model, norm='Linf', eps=8/255, version='standard')
    blind_robust_acc = evaluate(cirt_model, testloader, attack=blind_attack)
    print(f"Preprocessor-Blind AutoAttack Accuracy: {blind_robust_acc:.2f}%")

    # --- 2. The NeurIPS Test (Fully Adaptive White-Box) ---
    print("--- Running Fully Adaptive White-Box Evaluation ---")
    adaptive_attack = AutoAttack(cirt_model, norm='Linf', eps=8/255, version='standard')
    adaptive_robust_acc = evaluate(cirt_model, testloader, attack=adaptive_attack)
    print(f"Fully Adaptive AutoAttack Accuracy: {adaptive_robust_acc:.2f}%")
