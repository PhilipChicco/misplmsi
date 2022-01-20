
import torch, os, copy
import numpy as np

from tqdm import tqdm
from torchvision import transforms

from models import get_model
from utils.misc import convert_state_dict

from evaluators.base_eval import BaseTester
from loaders import  get_milfolder_test
from results_utils.collect_results import get_metrics
from evaluators.utils import compute_features, compute_tsne, draw_tsne, compute_features_tsne_only
from utils.misc import AverageMeter


class BaseMILTest(BaseTester):

    def __init__(self, cfg, logdir, logger):
        super().__init__(cfg, logdir, logger)

        # setup datasets
        print('Setting up data ...')
        self.data_transforms = {
            'test': transforms.Compose([
                
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        self.nclass = self.cfg['arch']['n_classes']
        self.test_dset, self.test_loader = get_milfolder_test(self.cfg, self.data_transforms)['test']


    def _run_test(self):
        
        self.inference_classification()

    def inference_classification(self):
        self.model.eval()
        val_accuracy = AverageMeter()

        with torch.no_grad():
            final_itr = tqdm(self.test_loader, ncols=80, desc='Inference (instance) ...')
            y_labels = []
            y_probs  = []
            y_preds  = []
            y_id     = []

            for i, (input, labels) in enumerate(final_itr):
                input  = input.to(self.device)
                y_labels.append(labels.data.cpu().numpy()[0])
                labels = labels.to(self.device)

                logits = self.model(input)

                preds  = self.model.module.pooling.predictions(logits)
                probs  = self.model.module.pooling.probabilities(logits)
                probs  = probs.detach()[:, 1].clone().data.cpu().numpy()[0]
                
                y_preds.append(preds.data.cpu().numpy()[0])
                y_probs.append(probs)
                y_id.append(i)

                accuracy = (preds == labels).sum().item() / labels.shape[0]
                val_accuracy.append(accuracy)

                final_itr.set_description('--- (test) | Accuracy: {:.5f}  :'.format(
                    val_accuracy.avg())
                )
            
        y_labels = np.array(y_labels)
        y_probs  = np.array(y_probs)
        y_preds  = np.array(y_preds)
        y_id     = np.array(y_id)
        
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

        # all predictions
        fp = open(os.path.join(self.logdir, 'predictions_0.5.csv'), 'w')
        fp.write('patch_id,target,pred,prob\n')
        for id_, target, prob in zip(y_id, y_labels, y_probs):
            fp.write('{},{},{},{:.3f}\n'.format(id_, target, int(prob >= 0.5), prob))
        fp.close()

        err = val_accuracy.avg()
        print('Accuracy: {:.5f} \n'.format(err))


    def _on_test_end(self):
        pass
        

    # not used
    def _on_test_start(self):
        pass

    def visualize_features(self):

        #self.inference_classification()
        
        print('----| Visualize features ....')

        cfg = self.cfg
        cfg['slide'] = None
        cfg['arch']['pooling'] = 'deepset_mean'
        cfg['trainer'] = ''
        self.embed_model = get_model(cfg)
        self.embed_model.features = torch.nn.Identity().to(self.device)
        self.ckpt = os.path.join(self.cfg['root'], self.cfg['testing']['tsne_model'])
        model_chk = torch.load(self.ckpt)
        state = convert_state_dict(model_chk["model_state"])
        self.embed_model.load_state_dict(state)
        self.embed_model = torch.nn.DataParallel(self.embed_model, device_ids=[0])
        self.embed_model.eval()
        print('Loaded ',self.ckpt)
        
        savepath = self.logdir
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        savename = os.path.join(savepath,'{}_TSNE_{}.png'.format(self.cfg['evaluator'],self.cfg['data']['test_nslides']))
        
        self.model.module.pooling.mode = 1

        # plot with images
        # fix the number of patches per slides in datasets MILFolder in preporcess
        # like this: grid[:k]
        X,y,y_img,y_pred = compute_features(self.model, self.cfg['training']['test_batch_size'], self.test_loader, embed=self.embed_model)
        print(X.shape, y.shape, y_img.shape)
        draw_tsne(X, y, y_img, y_pred, color_edge=True, save_path=savename, title=' ')
        print('Done !!!!')

        # TSNE only for all patches only
        #compute_features_tsne_only(self.model, self.cfg['training']['test_batch_size'], self.test_loader, embed=self.embed_model,
        #color_edge=True, save_path=savename, title=' ')




