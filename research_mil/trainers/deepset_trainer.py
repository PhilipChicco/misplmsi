
import torch,  os
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms

from trainers.base_trainer import BaseTrainer
from loaders import get_milrnnfolder
from utils.mil import calc_err
from utils.misc import AverageMeter
import loaders.utils_augmentation as utils_aug
from models import get_model
from utils.misc import convert_state_dict


class DeepSetTrain(BaseTrainer):

    def __init__(self, cfg, writer, logger):
        super().__init__(cfg, writer, logger)

        # setup datasets
        print('Setting up data ...')
        self.data_transforms = {
            'train': transforms.Compose([
                # with color aug twostep_color aug
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
                utils_aug.RandomRotate(),
                utils_aug.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        }

        loaders_dict = get_milrnnfolder(self.cfg, self.data_transforms)
        self.train_dset, self.train_loader = loaders_dict['train']
        self.val_dset,   self.val_loader   = loaders_dict['val']

        # Load feature (encoding model)
        cfg = self.cfg
        cfg['arch']['pooling'] = 'twostageavg'
        self.embed_model = get_model(cfg)
        self.ckpt = os.path.join(self.cfg['root'], self.cfg['testing']['feature'])
        model_chk = torch.load(self.ckpt)
        state = convert_state_dict(model_chk["model_state"])
        self.embed_model.load_state_dict(state)
        self.embed_model = torch.nn.DataParallel(self.embed_model, device_ids=[0])
        self.embed_model.eval()
        print('Loaded | ', self.ckpt)

        self.model.module.features = torch.nn.Identity().to(self.device)
        finetuned_params = list(self.model.module.features.parameters())
        new_params = [p for n, p in self.model.module.named_parameters()
                      if not n.startswith('features.')]

        param_groups = [{'params': finetuned_params, 'lr': self.cfg['training']['lr']},
                        {'params': new_params, 'lr': 0.01}]

        self.optimizer = optim.Adam(param_groups, weight_decay=1e-4)

        print()
        print(self.model)
        print()

    
    def _train_epoch(self, epoch):
        
        logits_losses  = AverageMeter()
        train_accuracy = AverageMeter()

        self.adjust_lr_staircase(
            self.optimizer.param_groups,
            [self.cfg['training']['lr'], 0.01],
            epoch + 1,
            [10,20,25,35],
            0.1
        )

        pbar = tqdm(self.train_loader, ncols=160, desc=' ')

        for i, data in enumerate(pbar):
            
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)

            self.optimizer.zero_grad()
            
            bs       = inputs.shape[0]
            k_size   = inputs.shape[1]
            inputs   = inputs.view((-1, 3, 256, 256))

            features = self.embed_model.module.features(inputs)
            #features = self.model.module.features(inputs)
            features = features.view(bs, k_size, 512, 8 , 8)
            logits   = self.model.module.pooling(features)

            loss  = self.model.module.pooling.loss(logits, labels)
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

        maxs = self.inference_aggregation()
        pred = [1 if x >= 0.5 else 0 for x in maxs]
        err, fpr, fnr = calc_err(pred, self.val_dset.targets)
        err = 1 - ((fpr + fnr) / 2.)

        # log
        self.writer.add_scalar('validation/accuracy', err, epoch)
        self.writer.add_scalar('validation/fpr', fpr, epoch)
        self.writer.add_scalar('validation/fnr', fnr, epoch)
        print()
        print('---> | accuracy : {:.4f}'.format(err))

        return {'acc': err,'fpr': fpr, 'fnr': fnr }
    
    def inference_aggregation(self):
        self.model.eval()
        probs = torch.FloatTensor(len(self.val_loader.dataset))

        with torch.no_grad():
            final_itr = tqdm(self.val_loader, ncols=80, desc='Inference (topk-Aggregation) ...')

            for i, data in enumerate(final_itr):
                inputs  = data[0]
                inputs  = inputs.to(self.device)

                bs = inputs.shape[0]
                k_size   = inputs.shape[1]
                inputs   = inputs.view((-1, 3, 256, 256))

                features = self.embed_model.module.features(inputs)
                features = features.view(bs, k_size, 512, 8 , 8)
                logits   = self.model.module.pooling(features)

                output   = self.model.module.pooling.probabilities(logits)
                probs[i] = output.detach()[:, 1].clone()

        return probs.cpu().numpy()

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

