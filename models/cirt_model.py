import torch
import torch.nn as nn
from robustbench.utils import load_model

class CIRT_Model(nn.Module):
    def __init__(self, num_classes=10, projection_dim=128):
        super().__init__()
        # Use the standard, non-robust WideResNet as the base
        self.backbone = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
        
        backbone_feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Linear(backbone_feature_dim, num_classes)
        
        self.disentanglement_head = nn.Sequential(
            nn.Linear(backbone_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x, with_projection=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if with_projection:
            projection = self.disentanglement_head(features)
            return logits, projection
        
        return logits