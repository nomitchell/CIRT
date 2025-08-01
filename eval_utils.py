import torch

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
