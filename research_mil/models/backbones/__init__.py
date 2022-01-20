
from .resnetv3 import resnet34 as resnet34v3


backbones = {
    'resnet34v3': resnet34v3,
}

def load_backbone(backbone, pretrained=True):
    net_names = list(backbones.keys())
    if backbone not in net_names:
        raise ValueError('Invalid choice for backbone - choices: {}'.format(' | '.join(net_names)))
    return backbones[backbone](pretrained)





