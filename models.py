from data import ReportData, BufferShuffler
import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import TfidfModel, LsiModel
from gensim.corpora.dictionary import Dictionary
import nltk
from nltk.corpus import stopwords
from scipy import spatial
from pathlib import Path
import pickle
import utils as ut
import pandas as pd
from networks import CNN, FFN, LSTM, NetworkTrainer
import nltk



class VocabularyEmbedder:
    def __init__(self, vocab_length=20000):
        """Vocabulary embedder, for use together with CNN when making our own word embeddings. 

        Args:
            vocab_length (int, optional): Vocabulary size. Defaults to 20000.
        """        
        self.vocab_length = vocab_length
        self.params = {'vocab_length': vocab_length}
        self.storage_format = 'words'
        self.unique_name = 'Vocab%s' % (self.vocab_length)
        self.vocabulary = None
        self.doc_length = 0

    def _prepare_doc(self, doc):
        """This is called by SummaryQualityModel, and determines how
        input data should be pre-processed for this embedder."""
        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    def _train(self, data, overwrite):
        """This is called by SummaryQualityModel, and trains the embedder.

        Args:
            data (ReportData): Pre-processed data (according to _prepare_doc) in format
                               (iterable[dict{'__key__', 'report.pyd', 'summary.pyd'}])
            overwrite (boolean): This argument is only used by Word2vecEmbedder. 
        """        
        print("\nMaking Vocabulary...")
        fdist = nltk.FreqDist()
        for element in data:
            self.doc_length = max(self.doc_length, len(element['report.pyd']), len(element['summary.pyd']))
            for word in element['report.pyd']:
                fdist[word] += 1
            for word in element['summary.pyd']:
                fdist[word] += 1
          
        common_words = fdist.most_common(self.vocab_length)

        self.vocabulary = {}
        for idx, word in enumerate(common_words):
            self.vocabulary[word[0]] = idx + 1

    def _embed(self, doc):
        """Called by SummaryQualityModel, determines how embedder should embed document. """
        idx_list = []
        for word in doc:
            if word in self.vocabulary.keys():
                idx_list.append(self.vocabulary[word])
        while len(idx_list) < self.doc_length:
            idx_list.append(0)
        return np.array(idx_list)

class Word2vecEmbedder:
    def __init__(self, dim=100, window=10, min_count=20, workers=4, epochs=50):
        """Word2vec embedder, for use together with CNN. 

        Args:
            dim (int, optional): Dimensionality of word embeddings. Defaults to 100.
            window (int, optional): Window size in Word2vec. Defaults to 10.
            min_count (int, optional): Number of times word must appear to be included in 
                                       vocabulary. Defaults to 20.
            workers (int, optional): Workers in training process. Defaults to 4.
            epochs (int, optional): Number of training epochs in Word2vec. Defaults to 50.
        """        
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.params = {}
        self.storage_format = 'words'
        self.unique_name = 'Word2vec%s_w%s_mc%s_e%s' % (dim, window, min_count, epochs)
        self.vocabulary = None
        self.doc_length = 0
        self.model = None

    def _prepare_doc(self, doc):
        """This is called by SummaryQualityModel, and determines how
        input data should be pre-processed for this embedder."""
        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    def _store_sentences(self, element):
        """For technical reasons, Word2vec must make its own training dataset. (see 
        documentation under Word2vecEmbedder._train(). This function is used by _train
        to determine how this training dataset should be made (list of tokenized sentences). """
        if isinstance(element, tuple):
            report = element[0]
            labels = element[1]
        else:
            report = element
        body = [[word.lower() for word in nltk.word_tokenize(sent, language='norwegian')] 
                 for sent in report.get_report_sentences()]
        summary = [[word.lower() for word in nltk.word_tokenize(sent, language='norwegian')] 
                 for sent in report.get_summary_sentences()]
        out = {'__key__': report.id, 'report.pyd': body, 'summary.pyd': summary}
        if isinstance(element, tuple):
            out['labels.pyd'] = labels
        return out
    
    def _get_list_of_docs(self, element):
        """Technical funtion, adds list of sentences in report and
        summary. When used together with 
        BufferShuffler, we get an iterable of all sentences, which
        is the appropriate input for Word2vec training with gensim. """
        return element['report.pyd'] + element['summary.pyd']

    def _apply(self, element):
        """Used in _train when using BufferShuffler. In this case, we have pre-processed
        data to appropriate format already, so no transformation is needed. """
        return element

    def _train(self, data, input_data, overwrite):
        """This is called by SummaryQualityModel, and trains the embedder.

        Args:
            data (ReportData): Pre-processed data (according to _prepare_doc) in format
                               (iterable[dict{'__key__', 'report.pyd', 'summary.pyd'}])
            overwrite (boolean): For technical reasons, the Word2vecEmbedder must make its own
                                 training dataset (it was difficult to implement such that 
                                 SummaryQualityModel did this for Word2vecEmbedder, which
                                 is the case for the other Embedder models). 
                                 Overwrite argument is then necessary to determine if 
                                 existing pre-processed dataset should be overwritten. 
        """     
        self.doc_length = 0
        print("\nFinding max length document...")
        for element in data:
            self.doc_length = max(self.doc_length, len(element['report.pyd']), len(element['summary.pyd']))

        dataname = data.path.split('_')[-1]
    
        path = 'data/restructured/sentences_words_%s' % dataname
        ReportData(path).create(data=input_data, overwrite=overwrite, apply=self._store_sentences)
        data = ReportData(path, apply=self._get_list_of_docs)
        data = BufferShuffler(data, self._apply, buffer_size=1000)

        print("\nBuilding Word2vec vocabulary")
        self.model = Word2Vec(size=self.dim, window=self.window, min_count=self.min_count,
                             workers=self.workers, sg=1, )
        self.model.build_vocab(data)

        print("\nTraining Word2vec model...")
        epochprinter = ut.EpochLogger()
        self.model.train(sentences=data, total_examples=self.model.corpus_count, 
                         epochs=self.epochs, callbacks=[epochprinter])

        self.params['emb_matrix'] =  self.model.wv
       

    def _embed(self, doc):
        """Called by SummaryQualityModel, determines how embedder should embed document. """
        idx_list = []

        for word in doc:
            if word in self.model.wv.vocab.keys():
                idx_list.append(self.model.wv.vocab[word].index+1)
        while len(idx_list) < self.doc_length:
            idx_list.append(0)

        return np.array(idx_list)

class TFIDFEmbedder:
    def __init__(self, no_below=15000, remove_stopwords=False):
        """TF-IDF embedder, which was tested as an alternative to LSA. 

        Args:
            no_below (int, optional): Number of documents a word must appear in for it to
                                      be included in the vocabulary. Defaults to 15000.
            remove_stopwords (bool, optional): Whether stopwords should be removed. 
                                               Defaults to False.
        """         
        self.dim = None
        self.params = None
        self.no_below = no_below
        self.remove_stopwords = remove_stopwords
        self.word_dict = None
        self.model = None
        self.storage_format = 'words'
        self.unique_name = 'TFIDF_nb%s_na%s_fmf%s_rs%s' % (self.no_below, self.remove_stopwords)

    def _prepare_doc(self, doc):
        """This is called by SummaryQualityModel, and determines how
        input data should be pre-processed for this embedder."""
        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    def _get_words(self, element):
        """This and _get_bow is used by BufferShuffler to transform input from pre-processed 
        format (list of words) to appropriate format for training using gensim. """
        if isinstance(element, dict):
            element = element['doc']
        if self.remove_stopwords:
            stopwords_list = stopwords.words('norwegian')
            doc = [word for word in element if word not in stopwords_list]
        else:
            doc = element
        return doc

    def _get_bow(self, element):
        """This and _get_words is used by BufferShuffler to transform input from pre-processed 
        format (list of words) to appropriate format for training using gensim. """
        element = self._get_words(element)
        element = self.word_dict.doc2bow(element)
        return self.tf_idf_model[element]

    def _train(self, data, input_data, overwrite):
        """This is called by SummaryQualityModel, and trains the embedder.

        Args:
            data (ReportData): Pre-processed data (according to _prepare_doc) in format
                               (iterable[dict{'__key__', 'report.pyd', 'summary.pyd'}])
            overwrite (boolean): This argument is only used by Word2vecEmbedder. 
        """     
        data_for_dict = BufferShuffler(data, self._get_words, buffer_size=1000)
        data_for_train = BufferShuffler(data, self._get_bow, buffer_size=1000)

        print("\nMaking LSA dictionary...")
        word_dict = Dictionary(data_for_dict)
        word_dict.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=None)
        word_dict.filter_n_most_frequent(self.fmf)
        self.word_dict = word_dict
        self.dim = len(word_dict)
        self.params = len(word_dict)

        print("\nTraining LSA model...")
        self.model = TfidfModel(dictionary=self.word_dict, smartirs='nfc')

    def _embed(self, doc):
        """Called by SummaryQualityModel, determines how embedder should embed document. """
        bow = self._get_bow(doc)
        z = np.zeros(self.dim)
        for element in bow:
            z[element[0]] = element[1]
        return z

class LSAEmbedder:
    def __init__(self, dim=500, no_below=15000, no_above=1, filter_most_frequent=0, 
                       remove_stopwords=False):
        """LSA embedder, to be used together with FFN or LSTM. 

        Args:
            dim (int, optional): Dimensionality of embeddings. Defaults to 500.
            no_below (int, optional): Number of documents a word must appear in for it to
                                      be included in the vocabulary. Defaults to 15000.
            no_above (int, optional): Maximum fraction of documents a word can be in for it
                                      to be included in the vocabulary. Defaults to 1.
            filter_most_frequent (int, optional): Most frequent words removed. Defaults to 0.
            remove_stopwords (bool, optional): Whether stopwords should be removed. 
                                               Defaults to False.
        """
        self.dim = dim
        self.params = dim
        self.no_below = no_below
        self.no_above = no_above
        self.fmf =  filter_most_frequent
        self.remove_stopwords = remove_stopwords
        self.word_dict = None
        self.tf_idf_model = None
        self.model = None
        self.storage_format = 'words'
        self.unique_name = 'LSA_dim%s_nb%s_na%s_fmf%s_rs%s' % (self.dim, self.no_below, self.no_above, 
                                                               self.fmf, self.remove_stopwords)

    def _prepare_doc(self, doc):
        """This is called by SummaryQualityModel, and determines how
        input data should be pre-processed."""
        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    def _get_words(self, element):
        """This and _get_bow is used by BufferShuffler to transform input from pre-processed 
        format (list of words) to appropriate format for training using gensim. """
        if isinstance(element, dict):
            element = element['doc']
        if self.remove_stopwords:
            stopwords_list = stopwords.words('norwegian')
            doc = [word for word in element if word not in stopwords_list]
        else:
            doc = element
        return doc

    def _get_bow(self, element):
        """This and _get_words is used by BufferShuffler to transform input from pre-processed 
        format (list of words) to appropriate format for training using gensim. """
        element = self._get_words(element)
        element = self.word_dict.doc2bow(element)
        return self.tf_idf_model[element]

    def _train(self, data, input_data, overwrite):
        """This is called by SummaryQualityModel, and trains the embedder.

        Args:
            data (ReportData): Pre-processed data (according to _prepare_doc) in format
                               (iterable[dict{'__key__', 'report.pyd', 'summary.pyd'}])
            overwrite (boolean): This argument is only used by Word2vecEmbedder. 
        """     
        data_for_dict = BufferShuffler(data, self._get_words, buffer_size=1000)
        data_for_train = BufferShuffler(data, self._get_bow, buffer_size=1000)

        print("\nMaking LSA dictionary...")
        word_dict = Dictionary(data_for_dict)
        word_dict.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=None)
        word_dict.filter_n_most_frequent(self.fmf)
        self.word_dict = word_dict

        print("\nTraining LSA model...")
        self.tf_idf_model = TfidfModel(dictionary=self.word_dict, smartirs='nfc')
        self.model = LsiModel(corpus=data_for_train, num_topics=self.dim, id2word=self.word_dict)

    def _embed(self, doc):
        """Called by SummaryQualityModel, determines how embedder should embed document. """
        bow = self._get_bow(doc)
        z = np.zeros(self.dim)
        for element in self.model[bow]:
            z[element[0]] = element[1]
        return z


class Doc2vecEmbedder:
    def __init__(self, dim=100, window=6, mc=20, workers=4, dm=0, epochs=50):
        """Doc2vec embedder, to be used together with FFN or LSTM. 

        Args:
            dim (int, optional): Dimensionality of doc embeddings. Defaults to 500.
            window (int, optional): Window size in Doc2vec. Defaults to 6.
            mc (int, optional): Number of times word must appear to be included in 
                                vocabulary. Defaults to 20.
            workers (int, optional): Workers in training process. Defaults to 4.
            dm (int, optional): Whether PV-DM version should be used. Defaults to 0 (PV-DBOW).
            epochs (int, optional): Number of training epochs in Doc2vec. Defaults to 50.
        """
        self.dim = dim
        self.params = dim
        self.window = window
        self.mc = mc
        self.workers = workers
        self.dm = dm
        self.epochs = epochs
        self.model = None
        self.storage_format = 'words'
        self.unique_name = 'Doc2vec_dim%s_w%s_mc%s_dm%s_e%s' % (self.dim, self.window, 
                                                                self.mc, self.dm, self.epochs)

    def _prepare_doc(self, doc):
        """This is called by SummaryQualityModel, and determines how
        input data should be pre-processed."""
        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    def _get_tagged_doc(self, element):
        """This is used by BufferShuffler to transform input from pre-processed 
        format (dict{'id':<any id>, 'doc':<words>}) to appropriate format for training 
        when using gensim. """
        return TaggedDocument(words=element['doc'], tags=[element['id']])

    def _train(self, data, input_data, overwrite):
        """This is called by SummaryQualityModel, and trains the embedder.

        Args:
            data (ReportData): Pre-processed data (according to _prepare_doc) in format
                               (iterable[dict{'__key__', 'report.pyd', 'summary.pyd'}])
            overwrite (boolean): This argument is only used by Word2vecEmbedder. 
        """     
        data = BufferShuffler(data, self._get_tagged_doc, buffer_size=1000)

        print("\nBuilding Doc2vec vocabulary")
        self.model = Doc2Vec(vector_size=self.dim, window=self.window, min_count=self.mc,
                             workers=self.workers, dm=self.dm)
        self.model.build_vocab(data)

        print("\nTraining Doc2vec model...")
        epochprinter = ut.EpochLogger()
        self.model.train(documents=data, total_examples=self.model.corpus_count, 
                         epochs=self.epochs, callbacks=[epochprinter])

    def _embed(self, doc):
        """Called by SummaryQualityModel, determines how embedder should embed document. """
        return self.model.infer_vector(doc)
 
class FFNModel:
    def __init__(self, layers=[100], batch_size=64, dropout=0.2, epochs=30, learning_rate=1e-4, 
                       tau_good=0.2, tau_bad=-0.2):
        """FFN summary quality model, for use together with TFIDF- LSA- or Doc2vecEmbedder.

        Args:
            layers (list, optional): Number of nodes to use in each layer. Number of layers 
                                becomes length of layers list. Defaults to [100].
            batch_size (int, optional): Batch size for training model. Defaults to 64.
            dropout (float, optional): Dropout rate when training model. Defaults to 0.2.
            epochs (int, optional): Number of epochs when training model. Defaults to 30.
            learning_rate (float, optional): Initial learning rate. Defaults to 1e-4.
            tau_good (float, optional): Tau_good to use for training model. Defaults to 0.2.
            tau_bad (float, optional): Tau_bad to use for training model. Defaults to -0.2.
        """
        self.name = 'FFN'
        self.storage_format = ''
        self.layers = layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.update_step_every = epochs // 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau_good = tau_good
        self.tau_bad = tau_bad

    def _get_documents(self, report):
        """Called by SummaryQualityModel. Determines what a document is for the embedder. 
        E.g. for FFN model, a document is the entire report/summary, while for LSTM model, 
        a document is a section/sentence. """
        return report.get_report_raw(), report.get_summary_raw()

    def _get_list_of_documents(self, element):
        """Called by SummaryQualitModel. Returns a list of all documents for each real 
        estate condition report. These will be unpacked by BufferShuffler when training
        embedder models. """
        doc1 = {'id': '%s_report' % element['__key__'], 'doc': element['report.pyd']}
        doc2 = {'id': '%s_summary' % element['__key__'], 'doc': element['summary.pyd']}
        return [doc1, doc2]

    def _train(self, train_path, val_path, emb_params):
        """Train neural network model on data in path. """
        model = FFN(emb_params, self.layers, self.dropout)
        self.trainer = NetworkTrainer(model, self.tau_good, self.tau_bad, self.batch_size, 
                                      self.learning_rate, self.update_step_every, self.epochs)
        self.model = self.trainer.train(train_path, val_path)

    def _embed(self, path, embedder, print_progress):
        """Use neural network model to embed data at path"""
        if self.trainer == None:
            raise RuntimeError("Model is not trained. Train model before embedding. ")
        return self.trainer.embed(path, print_progress)


class LSTMModel:
    def __init__(self, lstm_dim=100, num_lstm=1, bi_dir=False, output_dim=100, batch_size=64, 
                       dropout=0, epochs=30, learning_rate=1e-3, tau_good=0.2, tau_bad=-0.2):
        """LSTM summary quality model, for use together with LSAEmbedder or Doc2vecEmbedder.
            
        Args:
            lstm_dim (int, optional): Dimensionality of LSTM cell. Defaults to 100.
            num_lstm (int, optional): Number of LSTM layers. Defaults to 1.
            bi_dir (boolean, optional): Whether bi-directional LSTM layers are employed or not. 
                              Defaults to False.
            output_dim (int, optional): Output dimensionality of final, fully 
                              connected linear layer. Defaults to 100.
            batch_size (int, optional): Batch size for training model. Defaults to 64.
            dropout (float, optional): Dropout rate for training model. Defaults to 0.
            epochs (int, optional): Number of epochs in training model. Defaults to 30.
            learning_rate (float, optional): Initial learning rate for training model. 
                                             Defaults to 1e-3.
            tau_good (float, optional): Tau_good to use for training model. Defaults to 0.2.
            tau_bad (float, optional): Tau_bad to use for training model. Defaults to -0.2.
        """
        self.name = 'LSTM'
        self.storage_format = 'sections_'
        self.lstm_dim = lstm_dim
        self.num_lstm = num_lstm
        self.bi_dir = bi_dir
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.update_step_every = epochs // 3
        self.tau_good = tau_good
        self.tau_bad = tau_bad
        self.trainer = None

    def _get_documents(self, report):
        """Called by SummaryQualityModel. Determines what a document is for the embedder. 
        E.g. for FFN model, a document is the entire report/summary, while for LSTM model, 
        a document is a section/sentence. """
        return report.get_sections(), report.get_summary_sentences()
        

    def _get_list_of_documents(self, element):
        """Called by SummaryQualitModel. Returns a list of all documents for each real 
        estate condition report. These will be unpacked by BufferShuffler when training
        embedder models. """
        doc1 = [{'id': '%s_report_%s' % (element['__key__'], i),
                 'doc': element['report.pyd'][i]} for i in range(len(element['report.pyd']))]
        doc2 = [{'id': '%s_summary_%s' % (element['__key__'], i),
                 'doc': element['summary.pyd'][i]} for i in range(len(element['summary.pyd']))]
        return doc1 + doc2

    def _collate_train(self, batch):
        """Used for PackedSequence stuff, convenient with different-length sequences
        in batches for LSTM. """
        batch_z_r = [item[0] for item in batch]
        batch_z_s = [item[1] for item in batch]
        z_r, z_s = pack_sequence(batch_z_r, enforce_sorted=False), pack_sequence(batch_z_s, enforce_sorted=False)
        labels = torch.cat([item[2].reshape(1, 2) for item in batch], dim=0)
        return z_r, z_s, labels

    def _train(self, train_path, val_path, emb_params):
        """Train neural network model on data in path. """
        model = LSTM(emb_params, self.lstm_dim, self.num_lstm, self.bi_dir,
                                 self.output_dim, self.dropout)
        self.trainer = NetworkTrainer(model, self.tau_good, self.tau_bad, self.batch_size, 
                                      self.learning_rate, self.update_step_every, self.epochs)
        self.model = self.trainer.train(train_path, val_path, collate=self._collate_train)
        
    def _collate_test(self, batch):
        """Used for PackedSequence stuff, convenient with different-length sequences
        in batches for LSTM. """
        batch_z_r = [item[1] for item in batch]
        batch_z_s = [item[2] for item in batch]
        z_r, z_s = pack_sequence(batch_z_r, enforce_sorted=False), pack_sequence(batch_z_s, enforce_sorted=False)
        keys = [item[0] for item in batch]
        return keys, z_r, z_s
        
    def _embed(self, path, embedder, print_progress):
        """Use neural network model to embed data at path"""
        if self.trainer == None:
            raise RuntimeError("Model is not trained. Train model before embedding. ")

        return self.trainer.embed(path, print_progress, collate=self._collate_test)

class CNNModel:
    def __init__(self, embedding_size=100, output_size=200, kernels=[2,3,5,7,10], batch_size=64, 
                       dropout=0.1, epochs=30, learning_rate=1e-2, tau_good=0.2, tau_bad=-0.2):
        """CNN summary quality model, for use together with VocabularyEmbedder or Word2vecEmbedder.

        Args:
            embedding_size (int, optional): Dimensionality of word embeddings in EmbLayer/Word2vec. 
                                  Defaults to 100.
            output_size (int, optional): Number of filters per filter size and nodes in final linear layer. 
                               Defaults to 200.
            kernels (list[int], optional): List of filter sizes. Total number of filters becomes
                                 len(kernels)*output_size. Defaults to [2,3,5,7,10].
            batch_size (int, optional): Batch size for training model. Defaults to 64.
            dropout (float, optional): Dropout rate for training model. Defaults to 0.1.
            epochs (int, optional): Number of epochs when training model. Defaults to 30.
            learning_rate (float, optional): Initial learning rate for training model. 
                                             Defaults to 0.001.
            tau_good (float, optional): Tau_good to use for training model. Defaults to 0.2.
            tau_bad (float, optional): Tau_bad to use for training model. Defaults to -0.2.
        """                       
        self.name = 'CNN'
        self.storage_format = ''
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.kernels = kernels
        self.batch_size = batch_size
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.update_step_every = epochs // 3
        self.tau_good = tau_good
        self.tau_bad = tau_bad
        self.trainer = None
        
    def _get_documents(self, report):
        """Called by SummaryQualityModel. Determines what a document is for the embedder. 
        E.g. for FFN model, a document is the entire report/summary, while for LSTM model, 
        a document is a section/sentence. """
        return report.get_report_raw(), report.get_summary_raw()

    def _get_list_of_documents(self, element):
        """Called by SummaryQualitModel. Returns a list of all documents for each real 
        estate condition report. These will be unpacked by BufferShuffler when training
        embedder models. """
        return element

    def _train(self, train_path, val_path, emb_params):
        """Train neural network model on data in path. """
        model = CNN(emb_params, self.embedding_size, self.output_size, self.kernels, self.dropout)
        self.trainer = NetworkTrainer(model, self.tau_good, self.tau_bad, self.batch_size, 
                                      self.learning_rate, self.update_step_every, self.epochs)
        self.model = self.trainer.train(train_path, val_path)

    def _embed(self, path, embedder, print_progress):
        """Use neural network model to embed data at path"""
        if self.trainer == None:
            raise RuntimeError("Model is not trained. Train model before embedding. ")

        return self.trainer.embed(path, print_progress)

class EmptyModel:
    """Empty model, for using LSA or Doc2vec alone as baseline models. """    
    def __init__(self):
        self.name = 'Baseline'
        self.storage_format = ''

    def _get_documents(self, report):
        """Called by SummaryQualityModel. Determines what a document is for the embedder. 
        E.g. for FFN model, a document is the entire report/summary, while for LSTM model, 
        a document is a section/sentence. """
        return report.get_report_raw(), report.get_summary_raw()

    def _get_list_of_documents(self, element):
        """Called by SummaryQualitModel. Returns a list of all documents for each real 
        estate condition report. These will be unpacked by BufferShuffler when training
        embedder models. """
        doc1 = {'id': '%s_report' % element['__key__'], 'doc': element['report.pyd']}
        doc2 = {'id': '%s_summary' % element['__key__'], 'doc': element['summary.pyd']}
        return [doc1, doc2]

    def _train(self, train, val, emb_params):
        """Train neural network model on data in path. """
        pass

    def _embed(self, path, embedder, print_progress):
        """Use neural network model to embed data at path"""
        data = ReportData(path, apply=False, print_progress=print_progress)
        for element in data:
            yield {'id': element['__key__'], 
                   'z_r': embedder._embed(element['report.pyd']), 
                   'z_s': embedder._embed(element['summary.pyd'])}
    

class SummaryQualityModel:
    def __init__(self, embedder, model=EmptyModel()):
        """General class for summary quality models. Take an embedder and an optional
        neural network model as input, and combines them accordingly. 

        Args:
            embedder (<any>Embedder): Embedder model to use in summary quality model.
            model (<any>Model, optional): Neural network model to use together with embedder in 
                                          summary quality model. Defaults to EmptyModel (baseline).
        """        
        self.embedder = embedder
        self.model = model
        self.storage_path = 'data/restructured/%s%s' % (model.storage_format, embedder.storage_format)
        self.embedded_path = 'models/%s/data/%s' % (model.name, embedder.unique_name)

    def _store_documents(self, element):
        """This function describes how data should be pre-processed. Calls on 
        model._get_documents to determine what the documents are (e.g. report and summary
        or report sections and summary sentences), and calls on embedder._prepare_doc 
        to determine how each document should be stored (e.g. raw string, list of words."""
        if isinstance(element, tuple):
            report = element[0]
            labels = element[1]
        else:
            report = element

        body_documents, summary_documents = self.model._get_documents(report)

        if isinstance(body_documents, list):
            body = [self.embedder._prepare_doc(doc) for doc in body_documents]
        else: 
            body = self.embedder._prepare_doc(body_documents)

        if isinstance(summary_documents, list):
            summary = [self.embedder._prepare_doc(doc) for doc in summary_documents]
        else: 
            summary = self.embedder._prepare_doc(summary_documents)

        out = {'__key__': report.id, 'report.pyd': body, 'summary.pyd': summary}
        if isinstance(element, tuple):
            out['labels.pyd'] = labels
        return out
        

    def _store_embeddings(self, element):
        """This function calls on embedder._embed for creating embeddings of the 
        pre-processed documents. Stores embeddings with/without labels as 
        torch tensors, such that it is easy to train neural networks. """
        out = {}
        out['__key__'] = element['__key__']

        if isinstance(element['report.pyd'][0], list):
            z_r = np.array([self.embedder._embed(doc) for doc in element['report.pyd']])
        else:
            z_r = self.embedder._embed(element['report.pyd'])
        out['report.pth'] = torch.from_numpy(z_r).float()

        if isinstance(element['summary.pyd'][0], list):
            z_s = np.array([self.embedder._embed(doc) for doc in element['summary.pyd']])
        else:
            z_s = self.embedder._embed(element['summary.pyd']) 
        out['summary.pth'] = torch.from_numpy(z_s).float()

        if 'labels.pyd' in element.keys():
            out['labels.pth'] = torch.from_numpy(np.asarray(element['labels.pyd'])).float()

        return out


    def _create_dataset(self, dataname, data, overwrite):
        """Creates pre-processed dataset of ReportData type. """
        data_path = '%s_%s' % (self.storage_path, dataname)
        ReportData(data_path).create(data=data, overwrite=overwrite, apply=self._store_documents)
        return data_path

    def _create_embedded_dataset(self, dataname, input_path, overwrite):
        """Creates embeddings (by embedder part of SummaryQualityModel) of ReportData type."""
        if self.model.name == 'Baseline':
            return input_path
        input_data = ReportData(input_path, apply=False)
        data_path = '%s_%s' % (self.embedded_path, dataname)  
        ReportData(data_path).create(data=input_data, overwrite=overwrite, apply=self._store_embeddings)
        return data_path

    def _train_embedder(self, data_path, input_data, overwrite):
        """Call to train the embedder. Creates appropriate dataset and calls for 
        embedder._train. """
        data = ReportData(data_path, apply=self.model._get_list_of_documents)
        self.embedder._train(data, input_data, overwrite)

    def prepare_data(self, dataname, data, overwrite=False, overwrite_emb=False, 
                           train_embedder=False):
        """Prepare for training neural network part of model. Will pre-process data (if not 
        already done), train embedder models (if train_embedder=True) and create embeddings for 
        input data (if not already done). These embeddings will be stored, such that neural 
        networks can be trained on them directly. 

        Args:
            dataname (str): A name for the data to be used. Will determine the path for saving
                            data. 
            data (iterable[SummaryReport]/LabelledReportData): Data to prepare for training/testing. 
            overwrite (bool, optional): Boolean indicator of whether existing pre-processed data
                                        should be overwritten. If False, the data at path will be
                                        used, even if it does not correspond to the input 'data'. 
                                        Defaults to False.
            overwrite_emb (bool, optional): Whether new embeddings should be made by embedder 
                                            model. Defaults to False.
            train_embedder (bool, optional): Whether embedder model should be trained.
                                             Defaults to False.

        Returns:
            str: Path to embedded data, ready for training neural network. 
        """
        if dataname == None:
            return None

        data_path = self._create_dataset(dataname, data, overwrite)
        if train_embedder:
            self._train_embedder(data_path, data, overwrite)
        embedded_data_path = self._create_embedded_dataset(dataname, data_path, overwrite_emb)
        return embedded_data_path

    def train(self, train_name, train_data, val_name=None, val_data=None, 
                    overwrite=False, overwrite_emb=True, train_embedder=True):
        """Train neural network part of data. Will also perform the actions in 
        self.prepare data, if any of the steps have not already been done. 

        Args:
            train_name (str): Name of training data. 
            train_data (LabelledReportData): Training dataset. 
            val_name (str, optional): Name of validation data. Defaults to None (no val set used).
            val_data (LabelledReportData, optional): Val dataset. Defaults to None.
            overwrite (bool, optional): Whether existing pre-processed data with name 
                                        'train_name'/'val_name' should be overwritten. 
                                        Defaults to False.
            overwrite_emb (bool, optional): Whether existing embeddings from embedder models
                                            with name 'train_name'/'val_name' should be 
                                            overwritten. Defaults to True.
            train_embedder (bool, optional): Whether embedder model should be trained. 
                                             Defaults to True.
        """
        train_path = self.prepare_data(train_name, train_data, overwrite=overwrite, overwrite_emb=overwrite_emb, train_embedder=train_embedder)
        val_path = self.prepare_data(val_name, val_data, overwrite=overwrite, overwrite_emb=overwrite_emb)
        self.model._train(train_path, val_path, emb_params=self.embedder.params)

    def embed(self, data_name, data, overwrite=False, print_progress=True, overwrite_emb=False):
        """Return memory-friendly generator for embedded reports and summaries into the conceptual  
        summary content space (see master thesis), ready for measuring quality (cossim). 

        Args:
            data_name (str): Name of data to embed.
            data (iterable[SummaryReport]/LabelledReportData): Data to create embeddings of. 
            overwrite (bool, optional): Whether existing pre-processed data with name 
                                        'data_name' should be overwritten. 
                                        Defaults to False.
            print_progress (bool, optional): Whether progress of embedding should be printed. 
                                             Defaults to True.
            overwrite_emb (bool, optional): Whether existing embeddings from embedder models
                                            with name 'data_name' should be 
                                            overwritten. Defaults to False.

        Returns:
            generator[dict{'id', 'z_r', 'z_s'}]: Return memory friendly generator with 
                                                 embedder reports and summaries. 
        """
        path = self.prepare_data(data_name, data, overwrite=overwrite, overwrite_emb=overwrite_emb)
        return self.model._embed(path, self.embedder, print_progress)

    def predict(self, data_name, data, overwrite=False, overwrite_emb=True):
        """Return predicted qualitites for input data. 

        Args:
            data_name (str): Name of data to predict quality of. 
            data (iterable[SummaryReport]/LabelledReportData): Data to predict. 
            overwrite (bool, optional): Whether existing pre-processed data with name 
                                        'data_name' should be overwritten. 
                                        Defaults to False.
            overwrite_emb (bool, optional): Whether existing embeddings from embedder models
                                            with data name 'data_name' should be 
                                            overwritten. Defaults to False.

        Returns:
            pd.Series: Predicted qualities of report summaries, with ids in index. 
        """
        data = self.embed(data_name, data, overwrite=overwrite, overwrite_emb=overwrite_emb)
        keys = []
        qualities = []

        print('\nPredicting quality of %s' % data_name)
        for element in data:
            keys.append(element['id'])
            qualities.append(1 - spatial.distance.cosine(element['z_r'], element['z_s']))

        return pd.Series(qualities, index=keys)

    def save(self, modelname='default'):
        """Save model, using pickle. Not well tested, current implementation might have trouble 
        with saving, especially when models are large. 

        Args:
            modelname (str, optional): Save to path based on modelname. Defaults to 'default'.
        """        
        path = 'models/%s/%s.pkl' % (self.model.name, modelname)
        Path(path).mkdir(exist_ok=True, parents=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, modelname='default'):
        """Load model, using pickle.

        Args:
            modelname (str, optional): Load from path with based on modelname. 
                                       Defaults to 'default'.

        Returns:
            SummaryQualityModel: self
        """        
        path = 'models/%s/%s.pkl' % (self.model.name, modelname)
        with open(path, 'rb') as f:
            self = pickle.load(f)
        return self




        