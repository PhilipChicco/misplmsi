
import torch
import numpy as np
import logging
import datetime
import os

from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from matplotlib import offsetbox
import seaborn as sns

from PIL import Image, ImageDraw
import torch.nn.functional as F


#sns.set_palette('muted')
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
# visualization
colors = ['#00FF00', '#FF0000', '#2ca02c', '#d62728']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def compute_features_top(model, batch, dataloader):
    model.eval()
    N = len(dataloader.dataset)

    with torch.no_grad():
        final_itr = tqdm(dataloader, desc='Extracting features ...')
        for i, data in enumerate(final_itr):
            input = data[0]
            input = torch.stack(input, 0).squeeze(1).cuda()
            input = input.to(device)
            label = data[1]

            aux = model(input)[1]
            aux = model.module.pooling.get_global(aux).unsqueeze(0)
            aux = aux.data.cpu().numpy()
            lbl = label.data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')
                labels   = np.zeros((N,), dtype='int')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * batch: (i + 1) * batch] = aux
                labels[i * batch: (i + 1) * batch] = lbl

            else:
                # special treatment for final batch
                features[i * batch:] = aux
                labels[i * batch:]   = lbl

    return features, labels


def compute_features(model, batch, dataloader, embed=None):
    model.eval()
    N = len(dataloader.dataset)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        final_itr = tqdm(dataloader, desc='Extracting features ...')
        for i, (input_tensor, label) in enumerate(final_itr):

            input_var = input_tensor.to(device)
            
            x_feat, x_pred = model(input_var)
            x_pred = model.module.pooling.predictions(x_pred).data.cpu().numpy()
            aux = F.normalize(embed.module.pooling.enc(x_feat).view(input_var.size(0),-1),dim=1)

            #aux = F.normalize(model(input_var),dim=1)
            aux = aux.data.view(input_var.shape[0], -1).cpu().numpy()
            lbl = label.data.cpu().numpy()
            #img = input_var.data.view(input_var.shape[0], -1).cpu().numpy()
            img  = input_var.data.cpu().squeeze(0).numpy()
            img  = std * img.transpose((1, 2, 0)) + mean
            img  = np.clip(img, 0, 1)
            img  = np.expand_dims(img.transpose((2, 1, 0)),0)
            img  = img.reshape((img.shape[0],-1))

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')
                labels      = np.zeros((N,), dtype='int')
                labels_pred = np.zeros((N,), dtype='int')
                images   = np.zeros((N, img.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * batch: (i + 1) * batch] = aux
                labels[i * batch: (i + 1) * batch] = lbl
                labels_pred[i * batch: (i + 1) * batch] = x_pred
                images[i * batch: (i + 1) * batch] = img
            else:
                # special treatment for final batch
                features[i * batch:] = aux
                labels[i * batch:]   = lbl
                labels_pred[i * batch:]  = x_pred
                images[i * batch:]   = img

    return features, labels, images, labels_pred


# TSNE Only
def compute_features_tsne_only(model, batch, dataloader, embed=None,
                color_edge=False, save_path=None, cls_lbl=['MSS','MSI'],
                n_class=2, title=' '):
    model.eval()
    N    = len(dataloader.dataset)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        final_itr = tqdm(dataloader, desc='Extracting features ...')
        for i, (input_tensor, label) in enumerate(final_itr):

            input_var = input_tensor.to(device)
            
            x_feat, logits = model(input_var)
            x_pred = model.module.pooling.predictions(logits).data.cpu().numpy()[0]

            if i == 0:
                features = np.zeros((N, 64), dtype='float32')
                labels   = np.zeros((N,), dtype='int')
            

            if x_pred == label.data.cpu().numpy()[0]:
                
                x_prob  = model.module.pooling.probabilities(logits)
                x_prob  = x_prob[:,x_pred].data.cpu().numpy()[0]
                #print(x_pred, label.data.cpu().numpy(), x_prob)
        
                #if  x_prob >= 0.95:
                aux = F.normalize(embed.module.pooling.enc(x_feat).view(input_var.size(0),-1),dim=1)
                aux = aux.data.view(input_var.shape[0], -1).cpu().numpy()
                lbl = label.data.cpu().numpy()

                aux = aux.astype('float32')
                if i < len(dataloader) - 1:
                    features[i * batch: (i + 1) * batch] = aux
                    labels[i * batch: (i + 1) * batch] = lbl
                else:
                    # special treatment for final batch
                    features[i * batch:] = aux
                    labels[i * batch:]   = lbl
                #else:
                #    continue
            else: 
                continue

            
        
        # tsne
        X, y = features, labels
        tsne_width, tsne_height, tsne_dim, max_tile_size, n_iter = 10000, 10000, 2, 200, 5000
        tsne_obj = TSNE(n_jobs=4, random_state=1337, n_components=tsne_dim, learning_rate=150, perplexity=30, n_iter=n_iter)
        
        tsne_features = tsne_obj.fit_transform(X)

        # set matplotlib figure size
        plt.figure(figsize=(16, 16))

        # create tsne distribute image
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
###               
                

def compute_tsne(X, y, n_class=2,
                 savepath=None,
                 xlim=(-50,50), ylim=(-50,50),
                 cls_lbl=['MSS','MSI'],
                 title=' '):

    n_iter = 5000
    tsne = TSNE(n_jobs=4, random_state=1337, learning_rate=150, perplexity=30, n_iter=n_iter)
    embs = tsne.fit_transform(X)

    fig = plt.figure(figsize=(8,8))
    for i in range(n_class):
        inds = np.where(y == i)[0]
        plt.scatter(embs[inds, 0], embs[inds, 1], color=colors[i], marker='*', s=30)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.axis('off')
    plt.legend(cls_lbl)
    plt.grid(False)
    plt.grid(b=None)
    plt.title(title)
    
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.savefig(savepath.replace('.png','.pdf'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

def compute_tsne_aux(X, y, y_images, n_class=2, savepath=None,
                     xlim=(-50,50), ylim=(-50,50),
                     cls_lbl=['Benign','Tumor'], title='Fully Supervised'):

    tsne = TSNE(n_jobs=4, random_state=1337)
    #X = PCA(n_components=100).fit_transform(X)
    embs = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(64, 64))
    for i in range(n_class):
        inds = np.where(y == i)[0]
        plt.scatter(embs[inds, 0], embs[inds, 1], alpha=0.5, color=colors[i], marker='*')
    # if xlim:
    #     plt.xlim(xlim[0], xlim[1])
    # if ylim:
    #     plt.ylim(ylim[0], ylim[1])
    plt.legend(cls_lbl)

    if y_images is not None:
        thumb_frac = 0.09
        min_dist_2 = (thumb_frac * max(embs.max(0) - embs.min(0))) ** 2
        shown_images = np.array([2 * embs.max(0)])
        for i in range(y_images.shape[0]):
            dist = np.sum((embs[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, embs[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(to_image(y_images[i]), cmap='jet'),
                embs[i])
            ax.add_artist(imagebox)

    plt.title(title)
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.savefig(savepath.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    else:
        plt.show()

    print('Done ....')


def draw_tsne(X, labels, images, labels_pred, color_edge=False, save_path=None, cls_lbl=['MSS','MSI'],
                 n_class=2, title=' '):

    # tsne
    tsne_width, tsne_height, tsne_dim, max_tile_size, n_iter = 10000, 10000, 2, 200, 5000
    tsne_obj = TSNE(n_jobs=4, random_state=1337, n_components=tsne_dim, learning_rate=150, perplexity=30, n_iter=n_iter)
    tsne_features = tsne_obj.fit_transform(X)

    # set matplotlib figure size
    plt.figure(figsize=(16, 16))

    # create tsne distribute image
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
    ###

    # add images
    full_image = Image.new('RGB', (tsne_width, tsne_height), color=(255, 255, 255))
    for x, y, image_arr, image_label, image_pred in zip(tx, ty, images, labels, labels_pred):
        tile     = to_image_pil(image_arr).convert('RGB')
        old_size = tile.size
        ratio = float(max_tile_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        tile = tile.resize(new_size, Image.ANTIALIAS)

        if color_edge:
            dr = ImageDraw.Draw(tile)
            
            if image_label == 0 and image_pred == 0:
                color = (0, 255, 0) # GREEN
            elif image_label == 1 and image_pred == 1:
                color = (255, 0, 0) # RED
            elif image_label == 0 and image_pred == 1:
                color = (0,0,255)   # BLUE (FP)
            elif image_label == 1 and image_pred == 0:
                color = (255,255,0) # YELLOW (FN)

            dr.rectangle((0, 0, tile.width - 1, tile.height - 1), width=20, outline=color)

        full_image.paste(tile, (int((tsne_width - max_tile_size) * x), int((tsne_height - max_tile_size) * y)), mask=tile.convert('RGBA'))

    # save tsne
    full_image = full_image.transpose(Image.ROTATE_270).transpose(Image.ROTATE_180)
    full_image.save(save_path)


def to_image_pil(tensor,size=256):
    from PIL import Image
    tensor = torch.from_numpy(tensor.reshape(3, 256, 256))
    grid   = make_grid(tensor, nrow=1, normalize=True, scale_each=True)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = (Image.fromarray(ndarr)).resize((size, size), Image.ANTIALIAS)
    return im



def to_image(tensor,size=64):
    from PIL import Image
    tensor = torch.from_numpy(tensor.reshape(3, 256, 256))
    grid   = make_grid(tensor, nrow=1, normalize=True, scale_each=True)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = (Image.fromarray(ndarr)).resize((size, size), Image.ANTIALIAS)
    return np.array(im)