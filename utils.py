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
    """
    Split input data into a train, val and test set. 

    :param labels: Complete set of labels to divide into train/val/test. 
    :param ratio: List of float, with relative size that train/val/test set should have. 
    :param seed: Random state to use for splitting. 

    :return: `df_train, df_val, df_test` as 3 pandas df with labels for different reports. 
    """
    probs_trainval, probs_test = train_test_split(labels, test_size=ratio[2], random_state=seed)
    probs_train, probs_val = train_test_split(probs_trainval, test_size=ratio[1]/(ratio[0]+ratio[1]), random_state=seed)
    return probs_train, probs_val, probs_test

def score_readability(sentences, words):
    """Function to score liks and ovr. """

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

def measure_quality(model, datapath):
    """
    Measure quality of reports in data at datapath, using the input model.

    :param model: Model to use for quality measurement. 
    :param datapath: Path to data in webdataset-format, that is to be measured. 

    :return: Pandas series with ids and quality from model. 
    """
    data = wds.Dataset(datapath).decode()

    print("Calculating distances for data...")
    index = 0
        
    keys = []
    distances = []
    for house in data:
        index += 1 
        if index % 100 == 0:
            print('\r%i' % index, end='')
        keys.append(house['__key__'])
        distances.append(model(report=house['report.pyd'], summary=house['summary.pyd']))

    print("\nDone!")

    return pd.Series(distances, index=keys)

def plot_labels(labels):

    plt.figure(figsize=(8,5))
    plt.rcParams.update({'font.size': 22})
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    plt.hist(labels['prob_good'], bins=50, density=True, facecolor='blue', alpha=0.5)
    plt.xlabel(r'$P_{\boldsymbol{\mathbf{\mu}}}(y=1 \: | \: \boldsymbol{\mathbf{\lambda}})$')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('plot/labeldensity.png')
    plt.show()






def plot_quality(quality, labels=False, name=False, title=False, lab=False, include_t=False, show=True, all=False, density=True, xlim=True):
    """
    Plot two densities, one for distance to own report, the other for distance to other reports.
    """ 

    if type(labels) == type(pd.DataFrame()):
        good_idx = labels[labels['prob_bad'] <= 0.1].index
        medium_idx = labels[(labels['prob_bad'] > 0.1) & (labels['prob_bad'] < 0.9)].index
        bad_idx = labels[labels['prob_bad'] >= 0.9].index
        good = quality[good_idx]
        medium = quality[medium_idx]
        bad = quality[bad_idx]


    # Plot histogram
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


    #plt.hist(medium, bins=100, density=True, facecolor='y', alpha=0.5, label="Medium")
   
    if title:
        plt.title(title)
    if xlim:
        plt.xlim(-1,1)
    plt.xlabel(r'$q(\boldsymbol{\mathbf{r}}, \boldsymbol{\mathbf{s}})$')
    plt.ylabel('Density')
    if lab:
        #plt.legend()
        plt.legend(framealpha=0.4, loc='upper left')

    plt.tight_layout()

    if name:
        plt.savefig('plot/%s' % name)
    if show:
        plt.show()

def best_margin_loss(quality, labels):
    
    best_margin = -1
    best_loss = np.inf
    for margin in np.linspace(-1, 1, 400):
        loss = noise_aware_cosine_loss(quality, labels, margin, margin)
        if loss < best_loss:
            best_loss = loss
            best_margin = margin

    return best_loss, best_margin

def func1(x, y, tau_good, tau_bad):
    if y==1:
        return [max(0, tau_good-i) for i in x]
    if y==-1:
        return [max(0, i-tau_bad) for i in x]
    return

def plot_loss(tau_good, tau_bad, save=False):
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
    """
    Calculates a noise aware cosine similarity loss.
    """
    labels['quality'] = quality
    good = np.clip(tau_good - labels['quality'], a_min=0, a_max=2)
    bad = np.clip(labels['quality'] - tau_bad, a_min=0, a_max=2)
    labels['loss'] = labels['prob_bad']*bad + labels['prob_good']*good
    return labels['loss'].mean()

def get_best_threshold(quality, labels):
    """Gets maximum accuracy threshold to use for classification. """
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
    """
    Calculates accuracy, recall, precision and F1-score on the most probable labels, by using a 
    "maximum accuracy" threshold for classifying
    """
    labels['quality'] = quality
    labels['y'] = (labels['prob_good'] > labels['prob_bad']).astype(int)
    labels['predicted_y'] = (labels['quality'] >= best_threshold).astype(int)
    labels = labels.dropna()
    accuracy = sum(labels['y'] == labels['predicted_y'])/len(labels['y'])
    precision, recall, f_one, support = precision_recall_fscore_support(labels['y'], labels['predicted_y'], average='binary')

    return accuracy, precision, recall, f_one


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        self.epoch += 1
        print("\nEpoch %i" % self.epoch)


class progress_bar:
    def __init__(self, iterable, length):
        self.iterator = iter(iterable)
        self.index = 0
        self.starttime = time.time()
        self.lasttime = self.starttime
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if time.time() > (self.lasttime + 1) or self.index == self.length:
            self._print_progress()
        return next(self.iterator)

    def __len__(self):
        return self.length

    def _print_progress(self):
        self.lasttime = time.time()
        minutes = (time.time()-self.starttime) // 60
        seconds = (time.time()-self.starttime) % 60
        if self.length != None:
            x = int(60*self.index/self.length)
            sys.stdout.write("[%s%s%s]  %i/%i  %02d:%02d \r" % ("="*x, '>'*int(60>x), "."*(60-x-1), self.index, self.length, minutes, seconds))  
        else:
            sys.stdout.write("%i  %02d:%02d \r" % (self.index, minutes, seconds))
        if self.index == self.length:
            sys.stdout.write('\n')
        sys.stdout.flush()


class ProgressBar:
    def __init__(self, iterable):
        self.iterable = iterable
        try:
            self.length = len(iterable)
        except:
            self.length = None

    def __iter__(self):
        return progress_bar(self.iterable, self.length)

    def __len__(self):
        return self.length


class buffer_randomizer:
    def __init__(self, iterable, length, buffer_size, func):
        self.iterator = iter(iterable)
        self.length = length
        self.buffer_size = buffer_size
        self.func = func
        self.complete = False
        self.index = 0
        self.buffer = []
        while len(self.buffer) < self.buffer_size:
            self.buffer.extend(self.func(next(self.iterator)))
            self.index += 1

    def __iter__(self):
        return self

    def __next__(self):
        if (self.complete == False) and (len(self.buffer) < self.buffer_size):
            try:
                self.buffer.extend(self.func(next(self.iterator)))
                self.index += 1
            except StopIteration:
                self.complete = True

        try:
            newindex = random.randint(0, (len(self.buffer)-1))
            to_return = self.buffer.pop(newindex)
        except:
            raise StopIteration
        
        return to_return

    def __len__(self):
        return self.length


class BufferShuffler:
    def __init__(self, iterable, buffer_size, func):
        self.iterable = iterable
        self.buffer_size = buffer_size
        self.func = func
        try:
            self.length = len(iterable)
        except:
            self.length = None

    def __iter__(self):
        return buffer_randomizer(self.iterable, self.length, self.buffer_size, self.func)

        