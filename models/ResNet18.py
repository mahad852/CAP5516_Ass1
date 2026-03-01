from torchvision.models import resnet18, ResNet18_Weights
import torch

class ResNet18(torch.nn.Module):
    def __init__(self, num_classes: int, load_pretrained_weights: bool = True):
        super(ResNet18, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if load_pretrained_weights else None
        self.model = resnet18(weights=weights)
        self.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.fc = torch.nn.Identity()

    def get_features(self, x):
        return self.model(x)
    
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x