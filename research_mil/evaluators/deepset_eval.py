
import torch, os, copy
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torchvision import transforms

from models import get_model
from utils.misc import convert_state_dict
from evaluators.base_eval import BaseTester
from loaders import get_milrnnfolder_test
from utils.mil import calc_err
from utils.misc import AverageMeter
from results_utils.collect_results import get_metrics



class DeepSetTest(BaseTester):

    def __init__(self, cfg, writer, logger):
        super().__init__(cfg, writer, logger)

        # setup datasets
        print('Setting up data ...')
        self.data_transforms = {
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # load feature model
        cfg = self.cfg
        cfg['slide'] = None
        cfg['arch']['pooling'] = 'twostageavg'
        cfg['trainer'] = ''
        self.embed_model = get_model(cfg)
        self.ckpt = os.path.join(self.cfg['root'], self.cfg['testing']['feature'])
        model_chk = torch.load(self.ckpt)
        state = convert_state_dict(model_chk["model_state"])
        self.embed_model.load_state_dict(state)
        self.embed_model = torch.nn.DataParallel(self.embed_model, device_ids=[0])
        self.embed_model.eval()
        print('Loaded | ', self.ckpt)

        # classic
        loaders_dict = get_milrnnfolder_test(cfg, self.data_transforms)

        self.test_dset, self.test_loader = loaders_dict['test']
        print(self.model)


    def _run_test(self):

        maxs = self.inference()
        fp = open(os.path.join(self.logdir, 'predictions_0.5.csv'), 'w')
        fp.write('file,target,prediction,probability\n')
        for name, target, prob in zip(self.test_dset.slidenames, self.test_dset.targets, maxs):
            fp.write('{},{},{},{:.3f}\n'.format(os.path.split(name)[-1], target, int(prob >= 0.50), prob))
        fp.close()

        # check accuracy
        pred = [1 if x >= 0.5 else 0 for x in maxs]
        err, fpr, fnr = calc_err(pred, self.test_dset.targets)
        err = 1 - ((fpr + fnr) / 2.)
        print('--- (test)[BaseMIL-0.50] | Accuracy: {:.3f} | FPR: {:.3f} | FNR: {:.3f} '.format(
            err, fpr, fnr
        ))


    def inference(self):
        self.model.eval()
        probs = torch.FloatTensor(len(self.test_loader.dataset))
        val_accuracy = AverageMeter()

        with torch.no_grad():
            final_itr = tqdm(self.test_loader, ncols=80, desc='Inference ...')

            y_labels = []
            y_probs  = []
            y_preds  = []

            for i, (inputs,labels,_)in enumerate(final_itr):
                inputs   = inputs.to(self.device)
                y_labels.append(labels.data.cpu().numpy()[0])
                labels   = labels.to(self.device)

                bs = inputs.shape[0]
                k_size   = inputs.shape[1]
                inputs   = inputs.view((-1, 3, 256, 256))

                
                features = self.embed_model.module.features(inputs)
                features = features.view(bs, k_size, 512, 8 , 8)
                
                logits   = self.model.module.pooling(features)
                output   = self.model.module.pooling.probabilities(logits)
                probs[i] = output.detach()[:, 1].clone()

                # to generate scores
                preds    = self.model.module.pooling.predictions(logits)
                probs_i  = output.detach()[:, 1].clone().data.cpu().numpy()[0]

                y_preds.append(preds.data.cpu().numpy()[0])
                y_probs.append(probs_i)

                accuracy = (preds == labels).sum().item() / labels.shape[0]
                val_accuracy.append(accuracy)
                final_itr.set_description('--- (test) | Accuracy: {:.5f}  :'.format(
                    val_accuracy.avg())
                )
        
            y_labels = np.array(y_labels)
            y_probs  = np.array(y_probs)
            y_preds  = np.array(y_preds)
            
            out = get_metrics(y_labels, y_preds, y_probs, title='Test', 
                        savepath=os.path.join(self.logdir,'scores.png'))

            fp = open(os.path.join(self.logdir, 'meanscores.csv'), 'w')
            fp.write('F1,Precision,Recall,Specificity,Sensitivity,Accuracy,AUC\n')
            m = out['Test']
            fp.write('{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                100 * m['f1'], 100 * m['prec'], 100 * m['rec'], 100 * m['spec'],
                100 * m['sens'], 100 * m['acc'], 100 * m['auc']
            ))
            fp.close()

            err = val_accuracy.avg()
            print('Accuracy: {:.5f} \n'.format(err))

        return probs.cpu().numpy()


    def _on_test_end(self):
        pass

    def _on_test_start(self):
        pass

    def visualize_features(self):
        import torch.nn.functional as F
        import matplotlib
        import matplotlib.pyplot as plt
        from MulticoreTSNE import MulticoreTSNE as TSNE
        from sklearn.decomposition import PCA
        from matplotlib import offsetbox
        import seaborn as sns
        import torch.nn.functional as F
        colors   = ['#00FF00', '#FF0000', '#2ca02c', '#d62728']
        n_class = 2
        cls_lbl = ['MSS','MSI']
        title   = ' '
        batch   = self.cfg['training']['test_batch_size']

        print('----| Visualize features ....')
        savepath = os.path.join(self.logdir,'visual')
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        save_path = os.path.join(savepath,'{}_TSNE_{}.png'.format(self.cfg['evaluator'],self.cfg['data']['test_nslides']))
        self.model.module.pooling.mode = 2

        self.model.eval()
        N = len(self.test_loader.dataset)

        with torch.no_grad():
            final_itr = tqdm(self.test_loader, desc='Extracting features ...')
            for i, (input_tensor, label,_) in enumerate(final_itr):

                input_var = input_tensor.to(self.device)
                
                bs       = input_var.shape[0]
                k_size   = input_var.shape[1]
                inputs   = input_var.view((-1, 3, 256, 256))
                feat     = self.embed_model.module.features(inputs)
                feat     = feat.view(bs, k_size, 512, 8 , 8)

                aux      = F.normalize(self.model(feat),dim=1)

                aux = aux.data.view(input_var.shape[0], -1).cpu().numpy()
                lbl = label.data.cpu().numpy()
                

                if i == 0:
                    features = np.zeros((N, aux.shape[1]), dtype='float32')
                    labels   = np.zeros((N,), dtype='int')

                aux = aux.astype('float32')
                if i < len(self.test_loader) - 1:
                    features[i * batch: (i + 1) * batch] = aux
                    labels[i * batch: (i + 1) * batch]   = lbl
                else:
                    # special treatment for final batch
                    features[i * batch:] = aux
                    labels[i * batch:]   = lbl


        # tsne
        X, y = features, labels
        tsne_width, tsne_height, tsne_dim, max_tile_size, n_iter = 10000, 10000, 2, 200, 5000
        tsne_obj = TSNE(n_jobs=4, random_state=1337, n_components=tsne_dim, learning_rate=150, perplexity=30, n_iter=n_iter)
        tsne_features = tsne_obj.fit_transform(X)

        # set matplotlib figure size
        plt.figure(figsize=(16, 16))

        tx, ty = tsne_features[:, 0], tsne_features[:, 1]
        max_tx, min_tx, max_ty, min_ty = np.max(tx), np.min(tx), np.max(ty), np.min(ty)
        tx = (tx - min_tx) / (max_tx - min_tx)
        ty = (ty - min_ty) / (max_ty - min_ty)

        ###
        for i in range(n_class):
            inds = np.where(labels == i)[0]
            plt.scatter(ty[inds], tx[inds], color=colors[i], s=80)
        
        plt.axis('off')
        plt.legend(cls_lbl)
        plt.grid(False)
        plt.grid(b=None)
        plt.title(title)
        plt.savefig(save_path.replace('.png','._feats.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
