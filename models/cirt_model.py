import torch
import torch.nn as nn
from robustbench.utils import load_model

class CIRT_Model(nn.Module):
    def __init__(self, num_classes=10, projection_dim=128):
        super().__init__()
        # 1. Backbone: A pre-trained WideResNet-28-10 from RobustBench is a strong start.
        # We will use the standard, non-robust version as our base to modify.
        self.backbone = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
        
        # Replace the final linear layer of the backbone with an Identity layer
        # to get access to the final feature vector.
        backbone_feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 2. Final Classifier (for the main task)
        self.classifier = nn.Linear(backbone_feature_dim, num_classes)
        
        # 3. Disentanglement Head (for the invariance loss)
        self.disentanglement_head = nn.Sequential(
            nn.Linear(backbone_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        # Main forward pass
        features = self.backbone(x)
        logits = self.classifier(features)
        
        # Get projection from the disentanglement head
        projection = self.disentanglement_head(features)
        
        return logits, projection
