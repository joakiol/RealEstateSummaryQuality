from snorkel.labeling import PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import webdataset as wds
from sklearn.model_selection import train_test_split
import pickle

from labeling_functions import LABELING_FUNCTIONS, prepare_data_for_labeling_functions

class GenerativeModel:
    """Class for weak supervision generative model, used for making labels. """
    def __init__(self):
        """Initialize"""
        self.labeling_functions = LABELING_FUNCTIONS
        self.applier = PandasLFApplier(lfs=self.labeling_functions)
        self.df = None
        self.lambda_matrix = None
        self.model = None

    def __call__(self, data):
        """
        Takes iterable of SummaryReport objects as input, returns a list of probabilistic labels.
        That is, this function both trains the model on the data, and then finds probabilities. 
        """
        self.train(data)
        return self.predict_training_set()

    def train(self, data):
        """
        Trains the model on the input data. 
        
        :param data: VenduData-object, containing data for training the generative model. 
        """
        self.df = self._prepare_for_labeling(data)
        self.lambda_matrix = self.applier.apply(df=self.df)
        self.model = LabelModel(cardinality=2, verbose=True)
        self.model.fit(L_train=self.lambda_matrix, n_epochs=500, log_freq=100, seed=123)

    def predict(self, data):
        """
        Fit model on data.

        :param data: VenduData-object, containing data for training the generative model. 

        :return: Probabilistic labels
        """
        if self.model == None:
            raise RuntimeError("Model must be trained before predict-method is called. ")
        df = self._prepare_for_labeling(data)
        lambda_matrix = self.applier.apply(df=df)
        probs = self.model.predict_proba(lambda_matrix)
        df, probs = filter_unlabeled_dataframe(X=df, y=probs, L=lambda_matrix)
        return pd.DataFrame({'prob_bad': probs[:,0], 'prob_good': probs[:,1]}, index=df.index)

    def predict_training_set(self):
        """
        Predict labels on the set that was used for training the model. Much faster than calling 
        predict(data=training_data), since it takes time to prepare dataset for labeling functions. 
        """
        if self.model == None:
            raise RuntimeError("Model must be trained before predict-method is called. ")
        df = self.df
        lambda_matrix = self.lambda_matrix
        probs = self.model.predict_proba(lambda_matrix)
        df, probs = filter_unlabeled_dataframe(X=df, y=probs, L=lambda_matrix)
        return pd.DataFrame({'prob_bad': probs[:,0], 'prob_good': probs[:,1]}, index=df.index)

    def analyse_training_set(self, latexpath=None):

        analysis = LFAnalysis(L=self.lambda_matrix, lfs=self.labeling_functions).lf_summary()
        print(analysis)
        if latexpath != None:
            with open('%s.txt' % latexpath, 'w') as f:
                for idx, line in analysis.iterrows():
                    if line.Polarity == [0]:
                        predicted = 'Bad'
                    else:
                        predicted = 'Good'
                    f.write(f"{line['j']+1} & {predicted} & {100*line['Coverage']:.1f} \% & {100*line['Overlaps']:.1f} \% & {100*line['Conflicts']:.1f} \% \\\ \n")


    def plot_training_labels(self, name=None):

        labels = self.predict_training_set()

        plt.figure(figsize=(8,5))
        plt.rcParams.update({'font.size': 22})
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
        plt.hist(labels['prob_good'], bins=50, density=True, facecolor='b', alpha=0.5)
        plt.xlabel(r'$p(y=1|\boldsymbol \Lambda )$')
        plt.ylabel('Density')
        plt.tight_layout()
        if name:
            plt.savefig('plot/%s.pdf' % name)
        plt.show()
        
    def _prepare_for_labeling(self, data):
        return prepare_data_for_labeling_functions(data)

    def save(self, modelname='default'):
        path = 'models/WeakSupervision/%s' % modelname
        if self.model != None:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.df.to_csv('%s/df.csv' % path)
            np.save('%s/lambda.npy' % path, self.lambda_matrix)
        else:
            raise ValueError("Model has not been trained. Train model before saving. ")

    def load(self, modelname='default'):
        path = 'models/WeakSupervision/%s' % modelname
        try:
            self.df = pd.read_csv('%s/df.csv' % path, index_col='id')
            self.lambda_matrix = np.load('%s/lambda.npy' % path)
            self.model = LabelModel(cardinality=2, verbose=True)
            self.model.fit(L_train=self.lambda_matrix, n_epochs=500, log_freq=100, seed=123)
            return self
        except:
            raise NameError("%s is not a pre-trained model" % modelname)