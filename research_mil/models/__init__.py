
# backbones
import torch.nn as nn
from collections import OrderedDict

from models.backbones import load_backbone
from models.pooling import load_pooling


def get_model(cfg):
    backbone  = cfg['arch']['backbone']
    pooling   = cfg['arch']['pooling']
    n_classes = cfg['arch']['n_classes']
    embed     = cfg['arch']['embedding']

    backbone     = load_backbone(backbone)
    out_channels = backbone.inplanes

    pooling = load_pooling(pooling, out_channels, n_classes, embed)

    # final model
    model = nn.Sequential(OrderedDict([
        ('features', backbone),
        ('pooling' , pooling)
    ]))

    return model


if __name__ == '__main__':
    import torch
    cfg = {
        'arch':  {
                'backbone'  : 'resnet34v3',
                'pooling'   : 'twostageavg',
                'n_classes' :  2,
                'embedding' :  512
        },
        'slide' : None,
        'training': {'k_train' :32, 'k': 32}
    }

    net = get_model(cfg)
    print(net)

    x = torch.randn(2,3,256,256)
    f = net(x)
    #
    print(x.shape, ' -> ', f.shape, '\n', sep='')


