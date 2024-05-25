import torch.nn as nn
from torch import Tensor
from torchvision.models import DenseNet121_Weights, densenet121


class BinaryDenseNetClassifier(nn.Module):
    def __init__(self) -> None:
        super(BinaryDenseNetClassifier, self).__init__()
        self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)

        fc = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 2))

        self.densenet.classifier = fc

    def forward(self, x: Tensor) -> Tensor:
        return self.densenet(x)  # type: ignore[no-any-return]
