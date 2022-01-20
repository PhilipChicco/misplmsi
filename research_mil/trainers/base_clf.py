
import torch, torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms

from trainers.base_trainer import BaseTrainer
from loaders import get_milfolder
from utils.misc import AverageMeter



class BaseMIL(BaseTrainer):

    def __init__(self, cfg, writer, logger):
        super().__init__(cfg, writer, logger)

        # setup datasets
        print('Setting up data ...')
        
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        }

        self.nclass = self.cfg['arch']['n_classes']

        # using patch folders
        loaders_dict = get_milfolder(self.cfg, self.data_transforms, patch=False, use_sampler=True)        

        self.train_dset, self.train_loader = loaders_dict['train']
        self.val_dset,   self.val_loader   = loaders_dict['val']

        finetuned_params = list(self.model.module.features.parameters())
        new_params = [p for n, p in self.model.module.named_parameters()
                      if not n.startswith('features.')]

        param_groups = [{'params': finetuned_params, 'lr': self.cfg['training']['lr']},
                        {'params': new_params, 'lr': 0.01}]

        self.optimizer = optim.Adam(param_groups, weight_decay=1e-4)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        print(self.model.module.pooling)
    

    def _train_epoch(self, epoch):
        
        logits_losses  = AverageMeter()
        train_accuracy = AverageMeter()

        self.adjust_lr_staircase(
            self.optimizer.param_groups,
            [self.cfg['training']['lr'], 0.01],
            epoch + 1,
            [5,10,15],
            0.1
        )

        pbar = tqdm(self.train_loader, ncols=160, desc=' ')

        for i, data in enumerate(pbar):

            inputs = data[0]
            labels = data[1]

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(inputs)

            loss  = self.criterion(logits, labels)
            preds = self.model.module.pooling.predictions(logits)

            accuracy = (preds == labels).sum().item() / labels.shape[0]
            loss_val = loss.item()
            logits_losses.append(loss_val)
            train_accuracy.append(accuracy)

            loss.backward()
            self.optimizer.step()

            pbar.set_description('--- (train) | Loss: {:.4f}  | Accuracy: {:.3f}  :'.format(
                logits_losses.avg(), train_accuracy.avg())
            )

        step = epoch + 1
        self.writer.add_scalar('training/loss', logits_losses.avg(), step)
        self.writer.add_scalar('training/accuracy', train_accuracy.avg(), step)
        print()

    def _valid_epoch(self, epoch):
        val_accuracy = AverageMeter()

        with torch.no_grad():
            final_itr = tqdm(self.val_loader, ncols=80, desc=' ')

            for i, data in enumerate(final_itr):
                inputs = data[0]
                labels = data[1]

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                preds  = self.model.module.pooling.predictions(logits)

                accuracy = (preds == labels).sum().item() / labels.shape[0]
                val_accuracy.append(accuracy)

                final_itr.set_description('--- (val) | Accuracy: {:.3f}  :'.format(
                    val_accuracy.avg())
                )

        err = val_accuracy.avg()
        # log
        self.writer.add_scalar('validation/accuracy', err, epoch)

        return {'acc': err }


    def find_index(self,seq, item):
        for i, x in enumerate(seq):
            if item == x:
                return i
        return -1

    def adjust_lr_staircase(self, param_groups, base_lrs, ep, decay_at_epochs=[1, 2], factor=0.1):
        """Multiplied by a factor at the BEGINNING of specified epochs. Different
        param groups specify their own base learning rates.

        Args:
          param_groups: a list of params
          base_lrs: starting learning rates, len(base_lrs) = len(param_groups)
          ep: current epoch, ep >= 1
          decay_at_epochs: a list or tuple; learning rates are multiplied by a factor
            at the BEGINNING of these epochs
          factor: a number in range (0, 1)

        Example:
          base_lrs = [0.1, 0.01]
          decay_at_epochs = [51, 101]
          factor = 0.1
          It means the learning rate starts at 0.1 for 1st param group
          (0.01 for 2nd param group) and is multiplied by 0.1 at the
          BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the
          BEGINNING of the 101'st epoch, then stays unchanged till the end of
          training.

        NOTE:
          It is meant to be called at the BEGINNING of an epoch.
        """
        assert len(base_lrs) == len(param_groups), \
            "You should specify base lr for each param group."
        assert ep >= 1, "Current epoch number should be >= 1"

        if ep not in decay_at_epochs:
            return

        ind = self.find_index(decay_at_epochs, ep)

        for i, (g, base_lr) in enumerate(zip(param_groups, base_lrs)):
            g['lr'] = base_lr * factor ** (ind + 1)
            print('===> Param group {}: lr adjusted to {:.10f}'.format(i, g['lr']).rstrip('0'))


    # not used
    def _on_train_start(self):
        pass

    def _on_train_end(self):
        pass

    def _on_valid_start(self):
        pass

    def _on_valid_end(self):
        pass