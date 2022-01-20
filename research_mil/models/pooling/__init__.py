
from .classic import TwoStageAverage
from .deepset import DeepSet


poolings = {    

    'twostageavg' : TwoStageAverage,

    # Sets
    'deepset_mean'  : DeepSet,
    'deepset_max'   : DeepSet,
    'deepset_sum'   : DeepSet,
}


def load_pooling(pooling, in_channels, num_classes, embed):
    if pooling in ['deepset_mean','deepset_max','deepset_sum']:
        pool = pooling.split('_')[-1]
        pooling_module = poolings[pooling](in_channels, num_classes, embed, pool=pool)
    else:
        pooling_module = poolings[pooling](in_channels, num_classes, embed)
    
    return pooling_module

