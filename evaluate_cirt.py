import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from robustbench.utils import load_model
from torchattacks import AutoAttack
from models.cirt_model import CIRT_Model
from PIL import Image
import requests
import os

def evaluate(model, loader, attack=None, device='cpu'):
    correct = 0
    total = 0
    model.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if attack:
            inputs = attack(inputs, targets)
        with torch.no_grad():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100. * correct / total

def train_probe(probe, trainloader, testloader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    for epoch in range(epochs):
        probe.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = probe(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return evaluate(probe, testloader, device=device)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cirt_model = CIRT_Model().to(device)
    cirt_model.load_state_dict(torch.load("cirt_final.pth"))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # --- Main Evaluation (Preprocessor-Blind and White-Box) ---
    print("--- Running Core Evaluation ---")
    # Preprocessor-Blind Test
    standard_model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf').to(device)
    blind_attack = AutoAttack(standard_model, norm='Linf', eps=8/255)
    blind_robust_acc = evaluate(cirt_model, testloader, attack=blind_attack, device=device)
    print(f"Preprocessor-Blind AutoAttack Accuracy: {blind_robust_acc:.2f}%")

    # Fully Adaptive White-Box Test
    adaptive_attack = AutoAttack(cirt_model, norm='Linf', eps=8/255)
    adaptive_robust_acc = evaluate(cirt_model, testloader, attack=adaptive_attack, device=device)
    print(f"Fully Adaptive AutoAttack Accuracy: {adaptive_robust_acc:.2f}%")

    # --- Advanced Validation: Feature Inversion ---
    print("--- Running Feature Inversion Test ---")
    if not os.path.exists('cat.jpg'):
        r = requests.get('https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg')
        with open('cat.jpg', 'wb') as f:
            f.write(r.content)
    if not os.path.exists('parrot.jpg'):
        r = requests.get('https://i.natgeofe.com/n/e9023720-9115-4349-817a-31546842798c/spixs-macaw-pair_3x2.jpg')
        with open('parrot.jpg', 'wb') as f:
            f.write(r.content)

    content_image = Image.open("cat.jpg").convert("RGB")
    style_image = Image.open("parrot.jpg").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    content_image = transform(content_image).unsqueeze(0).to(device)
    style_image = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_features = cirt_model.backbone(content_image)
        _, style_projection = cirt_model(style_image, with_projection=True)

    gen_image = torch.randn_like(content_image, requires_grad=True)
    optimizer = torch.optim.Adam([gen_image], lr=0.1)

    for i in range(200):
        optimizer.zero_grad()
        gen_features = cirt_model.backbone(gen_image)
        _, gen_projection = cirt_model(gen_image, with_projection=True)
        content_loss = nn.functional.mse_loss(gen_features, content_features)
        style_loss = nn.functional.mse_loss(gen_projection, style_projection)
        (content_loss + style_loss).backward()
        optimizer.step()
    torchvision.utils.save_image(gen_image, "inversion_result.png")

    # --- Advanced Validation: Predictive Power Probes ---
    print("--- Running Predictive Power Test ---")
    cirt_model.eval().requires_grad_(False)

    # Extract features and projections
    all_features = []
    all_projections = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            features = cirt_model.backbone(inputs)
            _, projections = cirt_model(inputs, with_projection=True)
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

    content_probe = nn.Linear(cirt_model.classifier.in_features, 10).to(device)
    style_probe = nn.Linear(cirt_model.disentanglement_head[2].out_features, 10).to(device)

    content_acc = train_probe(content_probe, feature_loader, feature_loader, 10, device)
    style_acc = train_probe(style_probe, projection_loader, projection_loader, 10, device)

    print(f"Content Probe Accuracy: {content_acc:.2f}%")
    print(f"Style Probe Accuracy: {style_acc:.2f}%")
