import webdataset as wds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from gensim.models.callbacks import CallbackAny2Vec
import time
import sys
import random


def train_val_test_split(labels, ratio=[0.8, 0.1, 0.1], seed=1):
    """Split input data into a train, val and test set. 

    Args:
        labels (pd.DataFrame): Labels with report ids in index. 
        ratio (list, optional): Size of train, val and test. Defaults to [0.8, 0.1, 0.1].
        seed (int, optional): Seed for shuffling. Defaults to 1.

    Returns:
        probs_train (pd.DataFrame): Dataframe with labels for training set. 
        probs_val (pd.DataFrame): Dataframe with labels for validation set. 
        probs_test (pd.DataFrame): Dataframe with labels for test set. 
    """    
    
    probs_trainval, probs_test = train_test_split(labels, test_size=ratio[2], random_state=seed)
    probs_train, probs_val = train_test_split(probs_trainval, test_size=ratio[1]/(ratio[0]+ratio[1]), random_state=seed)
    return probs_train, probs_val, probs_test

def score_readability(sentences, words):
    """Function to score liks and ovr.

    Args:
        sentences (list[str]): List of sentences in text to score. 
        words (list[str]): List of words in text to score

    Returns:
        liks (float): LIKS score for input text (represented as list of sentences and words)
        ovr (float): OVR score for input text (represented as list of sentences and words)
    """    
    words = [word for word in words if word not in ('.', ',')]

    word_count = len(words)
    unique_words = len(set(words))
    sent_count = len(sentences)

    long_words = 0
    for word in words:
        if len(word) > 6:
            long_words += 1

    if word_count < 5:
        return 0, 0

    liks = 100 * (long_words / word_count) + (word_count / sent_count)
    ovr = 100 * np.log(unique_words) / np.log(word_count)

    return liks, ovr

def plot_quality(quality, labels=False, name=False, title=False, lab=False, show=True, all=False, density=True, xlim=True):
    """Plot two distributions, one for good summaries, and one for bad.

    Args:
        quality (list[float]): Quality measures to plot. 
        labels (bool, optional): Labels to use for determining which are good and bad summaries. 
                                 Defaults to False, in which case only one distribution are shown.
        name (str, optional): Save name. Defaults to False (no saving).
        title (str, optional): Title. Defaults to False (No title).
        lab (bool, optional): Whether labels should be shown in plot. Defaults to False.
        show (bool, optional): Whether to show plot. Defaults to True.
        all (bool, optional): Whether distribution of all should be shown in grey. 
                              Defaults to False.
        density (bool, optional): Whether normalization of distributions should be performed. 
                                  Defaults to True.
        xlim (bool, optional): Whether plots should be shown on [-1, 1]. Defaults to True.
    """    
    if type(labels) == type(pd.DataFrame()):
        good_idx = labels[labels['prob_bad'] <= 0.1].index
        medium_idx = labels[(labels['prob_bad'] > 0.1) & (labels['prob_bad'] < 0.9)].index
        bad_idx = labels[labels['prob_bad'] >= 0.9].index
        good = quality[good_idx]
        medium = quality[medium_idx]
        bad = quality[bad_idx]

    plt.figure(figsize=(8,5))
    plt.rcParams.update({'font.size': 22})
    #plt.rcParams.update({'font.size': 16})
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    if type(labels) == type(pd.DataFrame()):  
        plt.hist(good, bins=50, density=density, facecolor='g', alpha=0.5, label=r'$P_{\boldsymbol{\mathbf{\mu}}}(y=1 \: | \: \boldsymbol{\mathbf{\lambda}}) \geq 0.9$')
        if all:
            plt.hist(quality, bins=50, density=density, facecolor='grey', alpha=0.3, label='all')
        plt.hist(bad, bins=50, density=density, facecolor='r', alpha=0.5, label=r'$P_{\boldsymbol{\mathbf{\mu}}}(y=1 \: | \: \boldsymbol{\mathbf{\lambda}}) \leq 0.1$')
    else:
        plt.hist(quality, bins=50, density=density, facecolor='blue', alpha=0.5)
   
    if title:
        plt.title(title)
    if xlim:
        plt.xlim(-1,1)
    plt.xlabel(r'$q(\boldsymbol{\mathbf{r}}, \boldsymbol{\mathbf{s}})$')
    plt.ylabel('Density')
    if lab:
        plt.legend()
        #plt.legend(framealpha=0.4, loc='upper left')
    plt.tight_layout()
    if name:
        plt.savefig('plot/%s' % name)
    if show:
        plt.show()


def func1(x, y, tau_good, tau_bad):
    """Cosine embedding loss"""    
    if y==1:
        return [max(0, tau_good-i) for i in x]
    if y==-1:
        return [max(0, i-tau_bad) for i in x]
    return

def plot_loss(tau_good, tau_bad, save=False):
    """Plot loss function

    Args:
        tau_good (float): Tau_good for plot. 
        tau_bad ([type]): Tau_bad for plot. 
        save (str, optional): Save name. Defaults to False (no saving).
    """    
    plt.figure(figsize=(8,5))
    plt.rcParams.update({'font.size': 26})
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    x = np.linspace(-1, 1, 100)
    plt.title(r'$\tau_{{\text{{good}}}}={}, \tau_{{\text{{bad}}}}={}$'.format(tau_good, tau_bad))
    plt.plot(x, func1(x, 1, tau_good, tau_bad), color='g', label=r'$y=1$')
    plt.plot(x, func1(x, -1, tau_good, tau_bad), color='r', label=r'$y=-1$')
    plt.xlabel(r'$\text{cos sim}(\boldsymbol{\mathbf{z}}_{\boldsymbol{\mathbf{r}}}, \boldsymbol{\mathbf{z}}_{\boldsymbol{\mathbf{s}}})$')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if save != False:
        plt.savefig(save)
    plt.show()

def noise_aware_cosine_loss(quality, labels, tau_good, tau_bad):
    """Calculates a noise aware cosine similarity loss for a set of qualities/labels.

    Args:
        quality (pd.Series): Predicted qualities, with ids on index. 
        labels (pd.DataFrame): Labels, with ids on index. 
        tau_good (float): Which tau_good to use for loss function. 
        tau_bad (float): Which tau_bad to use for loss function. 

    Returns:
        float: Noise aware cosine embedding loss for input qualities (with labels). 
    """    
    labels['quality'] = quality
    good = np.clip(tau_good - labels['quality'], a_min=0, a_max=2)
    bad = np.clip(labels['quality'] - tau_bad, a_min=0, a_max=2)
    labels['loss'] = labels['prob_bad']*bad + labels['prob_good']*good
    return labels['loss'].mean()

def get_best_threshold(quality, labels):
    """Get maximum-accuracy threshold to use for classification.

    Args:
        quality (pd.Series): Predicted qualities, with ids on index. 
        labels (pd.DataFrame): Labels, with ids on index. 

    Returns:
        float: Threshold that maximizes accuracy. 
    """    
    labels['quality'] = quality
    labels['y'] = (labels['prob_good'] > labels['prob_bad']).astype(int)

    best_threshold = -1
    best_accuracy = 0

    for threshold in np.linspace(-1, 1, 400):
        labels['predicted_y'] = (labels['quality'] > threshold).astype(int)
        accuracy = sum(labels['y'] == labels['predicted_y'])/len(labels['y'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold

def classification_scores(quality, labels, best_threshold):
    """Calculates accuracy, recall, precision and F1-score on the labels, by using a 
    "maximum accuracy" threshold for classifying

    Args:
        quality (pd.Series): Predicted qualities, with ids on index. 
        labels (pd.DataFrame): Labels, with ids on index. 
        best_threshold (float): Threshold to use for classifying good/bad summaries. 

    Returns:
        accuracy (float): Accuracy for given qualities and threshold. 
        precision (float): Precision for given qualities and threshold. 
        recall (float): Recall for given qualities and threshold. 
        f_one (float): F1_score for given qualities and threshold. 
    """    
    labels['quality'] = quality
    labels['y'] = (labels['prob_good'] > labels['prob_bad']).astype(int)
    labels['predicted_y'] = (labels['quality'] >= best_threshold).astype(int)
    labels = labels.dropna()
    accuracy = sum(labels['y'] == labels['predicted_y'])/len(labels['y'])
    precision, recall, f_one, support = precision_recall_fscore_support(labels['y'], labels['predicted_y'], average='binary')

    return accuracy, precision, recall, f_one


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        """Callback for logging number of epochs in Word2vec/Doc2vec."""        
        self.epoch = 0

    def on_epoch_begin(self, model):
        self.epoch += 1
        print("\nEpoch %i" % self.epoch)
        