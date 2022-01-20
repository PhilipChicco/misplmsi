import os, sys, argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': ' DejaVu Sans', 'serif':['Computer Modern'], 'weight': 'bold'})
#rc('text', usetex=True)



def df_to_numpy(out):
    y_prob  = []
    y_label = []
    y_pred  = []

    for x in out:
        # ignore x[0] and x[1]: index and name
        y_label.append(x[2])
        y_pred.append(x[3])
        y_prob.append(x[4])

    return np.asarray(y_label), np.asarray(y_pred), np.asarray(y_prob)

def get_metrics(y_label, y_pred, y_probs, title=" ",
                labels=['N','T'], savepath="./"):

    f1   = f1_score(y_label, y_pred, labels=np.unique(y_label), average='weighted')
    prec = precision_score(y_label, y_pred, average='weighted')
    rec  = recall_score(y_label, y_pred, average='weighted')
    acc  = accuracy_score(y_label, y_pred)
    auc  = roc_auc_score(y_label, y_probs)

    cm1      = confusion_matrix(y_label, y_pred)
    sensitiv = cm1[0,0]/(cm1[0,0] + cm1[0,1])
    spec     = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

    fig, ax = plot_confusion_matrix(conf_mat=np.array(cm1),colorbar=False,show_absolute=True,
                                    show_normed=True,class_names=labels,
                                    figsize=(8.0,8.0))
    plt.title(title)
    plt.savefig(savepath, dpi=300)
    plt.close()
    plt.clf()

    return { title: {
        'f1': f1, 'prec': prec, 'rec': rec, 'spec': spec, 'sens': sensitiv,
        'acc': acc, 'auc': auc
    }}

def plot_roc_curve(label, prob, auc, name, color='orange', lw=2):

    # Equal Error Rate
    fpr, tpr, _ =roc_curve(label, prob)
    eer = brentq(lambda x: 1. - x - interp1d(fpr,tpr)(x), 0. ,1.)
    labelx = "({}, AUC $=$ {:.2f}, EER $=$ {:.2f})".format(name, auc, eer)
    plt.plot(fpr, tpr,color=color,lw=lw, label=str(labelx))

def plot_all_curves(labels, probs, aucs, names, colors, savepath):
    """

    Args:
        fpr_tpr: list of tuples of fpr,tpr for different methods
        aucs: aucs of methods
        names: the names of each method
        colors: colors to use
        savepath: the path at which file will be saved

    Returns: a complete plot

    """
    assert len(colors) == len(names)

    plt.figure(figsize=(5,5))

    for lbl, prob, auc, name, color in zip(labels, probs, aucs, names, colors):
        plot_roc_curve(lbl, prob, auc, name, color)

    plt.plot([0,1],[1, 0], color='navy', lw=1, linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(savepath,dpi=300)
    plt.close()

def summary_table(cfg, methods):

    fp = open(os.path.join(cfg.save, 'all_results.csv'),'w')
    fp.write('Method,F1,Precision,Recall,Specificity,Sensitivity,Accuracy,AUC\n')

    for met in methods:
        m = methods[met][met]
        fp.write('{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(met,
            100 * m['f1'], 100 * m['prec'], 100 * m['rec'], 100 * m['spec'],
            100 * m['sens'], 100 * m['acc'], 100 * m['auc']
        ))
    fp.close()



def main_amcwsi(cfg):
    if not os.path.exists(cfg.save):
        os.makedirs(cfg.save)

    mil_05 = pd.read_csv(os.path.join(cfg.root, 'predictions_0.50_data3.csv'))
    mil_25 = pd.read_csv(os.path.join(cfg.root, 'predictions_0.25_data3.csv'))
    mil_75 = pd.read_csv(os.path.join(cfg.root, 'predictions_0.75_data3.csv'))
    mil_90 = pd.read_csv(os.path.join(cfg.root, 'predictions_0.90_data3.csv'))

    mil_05_label, mil_05_pred, mil_05_prob = df_to_numpy(mil_05.to_records())
    mil_25_label, mil_25_pred, mil_25_prob = df_to_numpy(mil_25.to_records())
    mil_75_label, mil_75_pred, mil_75_prob = df_to_numpy(mil_75.to_records())
    mil_90_label, mil_90_pred, mil_90_prob = df_to_numpy(mil_90.to_records())

    mil_05_dict = get_metrics(mil_05_label, mil_05_pred, mil_05_prob, title="05", savepath=os.path.join(cfg.save, '05_cm.png'))
    mil_25_dict = get_metrics(mil_25_label, mil_25_pred, mil_25_prob, title="25", savepath=os.path.join(cfg.save, '25_cm.png'))
    mil_75_dict = get_metrics(mil_75_label, mil_75_pred, mil_75_prob, title="75", savepath=os.path.join(cfg.save, '75_cm.png'))
    mil_90_dict = get_metrics(mil_90_label, mil_90_pred, mil_90_prob, title="90", savepath=os.path.join(cfg.save, '90_cm.png'))



    labels  = [mil_05_label, mil_25_label, mil_75_label, mil_90_label]
    probs   = [mil_05_prob,  mil_25_prob,  mil_75_prob,  mil_90_prob]
    aucs    = [mil_05_dict['05']['auc'], mil_25_dict['25']['auc'], mil_75_dict['75']['auc'], mil_90_dict['90']['auc']]
    names   = ['05', '25','75','90' ]
    colors  = ['green', 'red', 'blue', 'yellow']
    savepth = os.path.join(cfg.save, 'amcwsi_asan_roc_curve.pdf')

    plot_all_curves(labels, probs, aucs, names, colors, savepth)
    plot_all_curves(labels, probs, aucs, names, colors, savepth.replace('.pdf', '.png'))
    plot_all_curves(labels, probs, aucs, names, colors, savepth.replace('.pdf', '.eps'))

    methods = {
        '05' : mil_05_dict,
        '25' : mil_25_dict,
        '75' : mil_75_dict,
        '90' : mil_90_dict,

    }
    summary_table(cfg, methods)



if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser('Evaluation Report Main Script')

    parent_parser.add_argument('--root', type=str, 
    default="./research_mil/results_utils/data/amcwsimay/amcwsi_data3/attention/full")
    parent_parser.add_argument('--save', type=str, 
    default="./research_mil/results_utils/out_270520/amcwsimay/amcwsi_data3/attention/full")

    main_amcwsi(parent_parser.parse_args())
