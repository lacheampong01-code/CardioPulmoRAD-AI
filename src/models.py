import torch
import torch.nn as nn
import torchvision.models as models

class TabularMLP(nn.Module):
    """Encodes 13 physiological vital signs and clinical variables."""
    def __init__(self, input_dim=13, output_dim=128):
        super(TabularMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.network(x)


class CardioPulmoRAD(nn.Module):
    """Multimodal Intermediate Fusion Framework for Cardiopulmonary Triage."""
    def __init__(self, num_classes=1):
        super(CardioPulmoRAD, self).__init__()
        
        # Vision Branch: Pre-trained DenseNet-121
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.vision_backbone = densenet.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Tabular Branch: 13 Clinical Vitals
        self.tabular_branch = TabularMLP(input_dim=13, output_dim=128)
        
        # Intermediate Fusion Layers (1024 + 128 = 1152-d)
        self.fusion_head = nn.Sequential(
            nn.Linear(1024 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, image, tabular_data):
        x_img = self.vision_backbone(image)
        x_img = self.global_pool(x_img)
        x_img = torch.flatten(x_img, 1)  # Tensor shape: [batch_size, 1024]
        
        x_tab = self.tabular_branch(tabular_data)  # Tensor shape: [batch_size, 128]
        
        v_fused = torch.cat((x_img, x_tab), dim=1)  # Tensor shape: [batch_size, 1152]
        
        risk_score = self.fusion_head(v_fused)
        return risk_score
