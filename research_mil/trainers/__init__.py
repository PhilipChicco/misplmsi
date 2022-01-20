
from .base_clf import BaseMIL
from .deepset_trainer import DeepSetTrain



trainers_dict = {
    # patch classifier
    'basemil' : BaseMIL,
    # DeepSet WSI Model
    'deepset' : DeepSetTrain,

}

def get_trainer(name):
    names = list(trainers_dict.keys())
    if name not in names:
        raise ValueError('Invalid choice for trainers - choices: {}'.format(' | '.join(names)))
    return trainers_dict[name]