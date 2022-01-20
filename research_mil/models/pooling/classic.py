import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification(nn.Module):
    def __init__(self, in_channels, classes, embed=None):
        super(Classification, self).__init__()

        self.classifier = nn.Linear(in_channels, classes)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.cross_entropy(logits, labels)


class TwoStageAverage(Classification):
    def __init__(self, in_channels, classes, embed):
        super(TwoStageAverage, self).__init__(in_channels, classes, embed)

        self.average  = nn.AdaptiveAvgPool2d((1,1))
        self.seg_mode = 0

    def forward(self, x):
    
        x   = self.average(x).flatten(1)
        out = self.classifier(x)

        if self.seg_mode == 1: 
            # pos class prob
            return F.softmax(out,1)[:,1].unsqueeze(1)

        return out

    