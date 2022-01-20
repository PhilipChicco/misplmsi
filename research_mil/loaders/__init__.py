import os, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from loaders.datasets import MILFolder, MILRNNFolder
from loaders.utils_augmentation import * 
from . import *

loaders = {
    "milFolder"      : MILFolder,
    "milRNNFolder"   : MILRNNFolder
}

# using folders
def get_milfolder(cfg, data_transforms, patch=False, use_sampler=True):
    state_shuff = True if patch else False
    print(f'Shuffle is set to {state_shuff}')
    print(f'Weighted-Sampler is set to {use_sampler}')
    class_map   = {x:idx for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    data_path   = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]

    t_dset = data_loader(
        root=data_path,
        split=cfg['data']['train_split'],
        transform=data_transforms['train'], 
        class_map=class_map, 
        nslides=cfg['data']['train_nslides'])
    
    if use_sampler:
        count_dict = get_class_distribution(cfg, t_dset)
        target_list = torch.tensor(t_dset.slideLBL)
        target_list = target_list[torch.randperm(len(target_list))]
        class_count   = [i for i in count_dict.values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        class_weights_all = class_weights[target_list]
        
        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        t_loader = DataLoader(t_dset,
                batch_size=cfg['training']['train_batch_size'],
                num_workers=cfg['training']['n_workers'],
                shuffle=False, # should be false for patch based training with sampler
                pin_memory=False, sampler=weighted_sampler
        )
    else:
        t_loader = DataLoader(t_dset,
                batch_size=cfg['training']['train_batch_size'],
                num_workers=cfg['training']['n_workers'],
                shuffle=state_shuff, # should be false for patch based training
                pin_memory=False,
        )

    v_dset = data_loader(
        root=data_path,
        split=cfg['data']['val_split'],
        transform=data_transforms['val'], class_map=class_map, nslides=cfg['data']['val_nslides'])
    v_loader = DataLoader(v_dset,
               batch_size=cfg['training']['val_batch_size'],
               num_workers=cfg['training']['n_workers'],
               shuffle=False, pin_memory=False
               )

    return {'train': (t_dset,t_loader), 'val': (v_dset,v_loader) }

def get_milfolder_test(cfg, data_transforms):

    class_map   = {x:idx for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    data_path   = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]

    t_dset = data_loader(
        root=data_path,
        split=cfg['data']['test_split'],
        transform=data_transforms['test'], 
        class_map=class_map, 
        nslides=cfg['data']['test_nslides'],
        train=False)
    t_loader = DataLoader(t_dset,
               batch_size=cfg['training']['test_batch_size'],
               num_workers=cfg['training']['n_workers']//2,
               shuffle=False, pin_memory=False, drop_last=False,
               )

    return {'test': (t_dset, t_loader) }

def get_milrnnfolder(cfg, data_transforms, patch=False):

    class_map = {x: idx for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    data_path = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]
    print(data_loader)

    t_dset = data_loader(
        root=data_path,
        split=cfg['data']['train_split'],
        transform=data_transforms['train'],
        s=cfg['training']['k_train'],
        shuffle=True,
        class_map=class_map, nslides=cfg['data']['train_nslides'])

    t_loader = DataLoader(t_dset,
        batch_size=cfg['training']['train_batch_size'],
        num_workers=cfg['training']['n_workers'],
        shuffle=True,
        pin_memory=False
    )

    v_dset = data_loader(
        root=data_path,
        split=cfg['data']['val_split'],
        transform=data_transforms['val'],
        s=cfg['training']['k_train'],
        shuffle=False,
        class_map=class_map, nslides=cfg['data']['val_nslides'])

    v_loader = DataLoader(v_dset,
        batch_size=cfg['training']['val_batch_size'],
        num_workers=cfg['training']['n_workers'],
        shuffle=False,
        pin_memory=False
    )

    return {'train': (t_dset, t_loader), 'val': (v_dset, v_loader)}

def get_milrnnfolder_test(cfg, data_transforms):
    class_map = {x: idx for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    data_path = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]

    t_dset = data_loader(
        root=data_path,
        split=cfg['data']['test_split'],
        transform=data_transforms['test'],
        s=cfg['training']['k'],
        shuffle=False,
        class_map=class_map, nslides=cfg['data']['test_nslides'],train=False)

    t_loader = DataLoader(t_dset,
                  batch_size=cfg['training']['test_batch_size'],
                  num_workers=cfg['training']['n_workers']//2,
                  shuffle=False,
                  pin_memory=False
                  )

    return {'test': (t_dset, t_loader)}

def get_class_distribution(cfg, dataset_obj):
    
    count_dict = {x: 0 for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    class_map  = {idx: x for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    
    for element in dataset_obj.slideLBL:
        y_lbl = class_map[element]
        count_dict[y_lbl] += 1  

    return count_dict

#
datamethods = {
    'milFolder'    : get_milfolder,
    'milRNNFolder' : get_milrnnfolder
}
