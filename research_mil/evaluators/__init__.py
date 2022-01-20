
# DeepSet
from .base_clf_test import BaseMILTest
from .deepset_eval import DeepSetTest


testers_dict ={
    # IN USE
    'basemil'       : BaseMILTest,
    'deepset'       : DeepSetTest,
}

def get_tester(name):
    names = list(testers_dict.keys())
    if name not in names:
        raise ValueError('Invalid choice for testers - choices: {}'.format(' | '.join(names)))
    return testers_dict[name]