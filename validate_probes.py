import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from models.cirt_model import CIRT_Model
import argparse
from tqdm import tqdm

def evaluate_probe(probe, loader):
    correct = 0
    total = 0
    probe.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = probe(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100. * correct / total

def train_probe(probe, trainloader, testloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(probe.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for epoch in range(epochs):
        probe.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = probe(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return evaluate_probe(probe, testloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIRT Probe Validation')
    parser.add_argument('--data-dir', type=str, default='./data', help='directory for dataset')
    parser.add_argument('--model-path', type=str, default='cirt_model.pth', help='path to trained CIRT model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Load CIRT model and freeze its weights
    cirt_model = CIRT_Model().to(device)
    cirt_model.load_state_dict(torch.load(args.model_path))
    cirt_model.eval()
    for param in cirt_model.parameters():
        param.requires_grad = False

    # Extract features and projections
    all_features = []
    all_projections = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            features = cirt_model.backbone(inputs)
            _, projections = cirt_model(inputs)
            all_features.append(features.cpu())
            all_projections.append(projections.cpu())
            all_labels.append(targets.cpu())

    all_features = torch.cat(all_features)
    all_projections = torch.cat(all_projections)
    all_labels = torch.cat(all_labels)

    feature_dataset = TensorDataset(all_features, all_labels)
    projection_dataset = TensorDataset(all_projections, all_labels)

    feature_loader = DataLoader(feature_dataset, batch_size=128, shuffle=True)
    projection_loader = DataLoader(projection_dataset, batch_size=128, shuffle=True)

    # 1. Create Probes
    backbone_feature_dim = cirt_model.backbone.linear.in_features
    projection_dim = 128
    content_probe = nn.Linear(backbone_feature_dim, 10).to(device)
    style_probe = nn.Linear(projection_dim, 10).to(device)

    # 2. Train and Evaluate Probes
    content_acc = train_probe(content_probe, feature_loader, feature_loader)
    style_acc = train_probe(style_probe, projection_loader, projection_loader)

    print(f"Content Probe Accuracy: {content_acc:.2f}%")
    print(f"Style Probe Accuracy: {style_acc:.2f}%")
