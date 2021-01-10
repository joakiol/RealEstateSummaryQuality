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
from networks import CNN, FFN, LSTM, cosine_loss, NetworkTrainer, Attn
import nltk



class VocabularyEmbedder:
    def __init__(self, vocab_length=10000):
        self.vocab_length = vocab_length
        self.params = {'vocab_length': vocab_length}
        self.storage_format = 'words'
        self.unique_name = 'Vocab%s' % (self.vocab_length)
        self.vocabulary = None
        self.doc_length = 0

    def _prepare_doc(self, doc):
        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    def train(self, data, input_data, overwrite):

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

    def embed(self, doc):
        idx_list = []

        for word in doc:
            if word in self.vocabulary.keys():
                idx_list.append(self.vocabulary[word])
        while len(idx_list) < self.doc_length:
            idx_list.append(0)
        return np.array(idx_list)

class Word2vecEmbedder:
    def __init__(self, dim=100, window=6, min_count=20, workers=4, epochs=50):
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
        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    def _store_sentences(self, element):
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
        return element['report.pyd'] + element['summary.pyd']

    def _apply(self, element):
        return element

    def train(self, data, input_data, overwrite):

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
       


    def embed(self, doc):

      
        idx_list = []

        for word in doc:
            if word in self.model.wv.vocab.keys():
                idx_list.append(self.model.wv.vocab[word].index+1)
        while len(idx_list) < self.doc_length:
            idx_list.append(0)

        return np.array(idx_list)

class TFIDFEmbedder:
    def __init__(self, dim=100, no_below=20, remove_stopwords=True):

        self.dim = dim
        self.params = dim
        self.no_below = no_below
        self.remove_stopwords = remove_stopwords
        self.word_dict = None
        self.model = None
        self.storage_format = 'words'
        self.unique_name = 'TFIDF_dim%s_nb%s_na%s_fmf%s_rs%s' % (self.dim, self.no_below, self.remove_stopwords)

    def _prepare_doc(self, doc):

        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]


    def _get_words(self, element):
        if isinstance(element, dict):
            element = element['doc']
        if self.remove_stopwords:
            stopwords_list = stopwords.words('norwegian')
            doc = [word for word in element if word not in stopwords_list]
        else:
            doc = element
        return doc

    def _get_bow(self, element):
        element = self._get_words(element)
        element = self.word_dict.doc2bow(element)
        return self.tf_idf_model[element]

    def train(self, data, input_data, overwrite):

        data_for_dict = BufferShuffler(data, self._get_words, buffer_size=1000)
        data_for_train = BufferShuffler(data, self._get_bow, buffer_size=1000)

        print("\nMaking LSA dictionary...")
        word_dict = Dictionary(data_for_dict)
        word_dict.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=None)
        word_dict.filter_n_most_frequent(self.fmf)
        self.word_dict = word_dict
        print(len(self.word_dict))

        print("\nTraining LSA model...")
        self.model = TfidfModel(dictionary=self.word_dict, smartirs='nfc')

    def embed(self, doc):

        bow = self._get_bow(doc)
        z = np.zeros(self.dim)
        for element in bow:
            z[element[0]] = element[1]
        return z

class LSAEmbedder:
    def __init__(self, dim=100, no_below=20, no_above=1, filter_most_frequent=0, remove_stopwords=True):

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

        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    # def _unpack_documents(self, element):
    #     return [element['report.pyd'], element['summary.pyd']]

    def _get_words(self, element):
        if isinstance(element, dict):
            element = element['doc']
        if self.remove_stopwords:
            stopwords_list = stopwords.words('norwegian')
            doc = [word for word in element if word not in stopwords_list]
        else:
            doc = element
        return doc

    def _get_bow(self, element):
        element = self._get_words(element)
        element = self.word_dict.doc2bow(element)
        return self.tf_idf_model[element]

    def train(self, data, input_data, overwrite):

        data_for_dict = BufferShuffler(data, self._get_words, buffer_size=1000)
        data_for_train = BufferShuffler(data, self._get_bow, buffer_size=1000)

        print("\nMaking LSA dictionary...")
        word_dict = Dictionary(data_for_dict)
        word_dict.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=None)
        word_dict.filter_n_most_frequent(self.fmf)
        self.word_dict = word_dict
        print(len(self.word_dict))

        print("\nTraining LSA model...")
        self.tf_idf_model = TfidfModel(dictionary=self.word_dict, smartirs='nfc')
        self.model = LsiModel(corpus=data_for_train, num_topics=self.dim, id2word=self.word_dict)

    def embed(self, doc):

        bow = self._get_bow(doc)
        z = np.zeros(self.dim)
        for element in self.model[bow]:
            z[element[0]] = element[1]
        return z

    # def embed_words(self, element):

    #     key = element['__key__']
    #     report_bow = self._get_bow(element['report.pyd'])
    #     summary_bow = self._get_bow(element['summary.pyd'])

    #     z_r = np.zeros(self.dim)
    #     for element in self.model[report_bow]:
    #         z_r[element[0]] = element[1]

    #     z_s = np.zeros(self.dim)
    #     for element in self.model[summary_bow]:
    #         z_s[element[0]] = element[1]

    #     return {'id': key, 'z_r': z_r, 'z_s': z_s}

    # def save(self, modelname='LSA%s' % self.dim):

    #     path = 'models/embedders/%s' % modelname
    #     self.word_dict.save('%s/word_dict' % path)
    #     self.tf_idf_model.save('%s/tfidf' % path)
    #     self.model.save('%s/lsamodel' % path)
    #     params = {'dim': self.dim, 'nb': self.no_below, 'na': self.no_above, 'fmf': self.fmf, 
    #               'rs': self.remove_stopwords}
    #     with open('%s/params.pkl' % path, 'wb') as f:
    #         pickle.dump(params), f)
        

    # def load(self, modelname='default'):

    #     path = 'models/embedders/%s' % modelname 
    #     try:
    #         with open('%s/params.pkl' % path, 'rb') as f:
    #             params = pickle.load(f)
    #         self.__init__(dim=params['dim'], no_below=params['nb'], no_above=params['na'], 
    #                       filter_most_frequent=params['fmf'], remove_stopwords=params['rs'])
    #         self.word_dict = Dictionary.load('%s/dict' % path)
    #         self.tf_idf_model = TfidfModel.load('%s/tfidf_model' % path)
    #         self.model = LsiModel.load('%s/model' % path)
    #     except:
    #         raise NameError("%s is not a pre-trained model" % modelname)

class Doc2vecEmbedder:
    def __init__(self, dim=100, window=6, mc=20, workers=4, dm=0, epochs=50):

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
        return [word.lower() for word in nltk.word_tokenize(doc, language='norwegian')]

    def _get_tagged_doc(self, element):
        return TaggedDocument(words=element['doc'], tags=[element['id']])

    def train(self, data, input_data, overwrite):

        data = BufferShuffler(data, self._get_tagged_doc, buffer_size=1000)

        print("\nBuilding Doc2vec vocabulary")
        self.model = Doc2Vec(vector_size=self.dim, window=self.window, min_count=self.mc,
                             workers=self.workers, dm=self.dm)
        self.model.build_vocab(data)

        print("\nTraining Doc2vec model...")
        epochprinter = ut.EpochLogger()
        self.model.train(documents=data, total_examples=self.model.corpus_count, 
                         epochs=self.epochs, callbacks=[epochprinter])

    def embed(self, doc):
        return self.model.infer_vector(doc)
 

    # def embed_words(self, element):
    #     key = element['__key__']
    #     z_r = self.model.infer_vector(element['report.pyd'])
    #     z_s = self.model.infer_vector(element['summary.pyd'])
    #     return {'id': key, 'z_r': z_r, 'z_s': z_s}

class LSTMModel:
    def __init__(self, doc_type='sections', lstm_dim=100, num_lstm=1, bi_dir=False, output_dim=100, batch_size=64, dropout=0.1, epochs=30, learning_rate=0.001, tau_good=0.2, tau_bad=-0.2):

        self.name = 'LSTM'
        self.doc_type = doc_type
        if doc_type == 'sections':
            self.storage_format = 'sections_'
        elif doc_type == 'sentences': 
            self.storage_format = 'sentences_'
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
        if self.doc_type == 'sections':
            return report.get_sections(), report.get_summary_sentences()
        elif self.doc_type == 'sentences':
            return report.get_report_sentences(), report.get_summary_sentences()

    def _get_list_of_documents(self, element):

        doc1 = [{'id': '%s_report_%s' % (element['__key__'], i),
                 'doc': element['report.pyd'][i]} for i in range(len(element['report.pyd']))]
        doc2 = [{'id': '%s_summary_%s' % (element['__key__'], i),
                 'doc': element['summary.pyd'][i]} for i in range(len(element['summary.pyd']))]
        return doc1 + doc2

    def _collate_train(self, batch):
        batch_z_r = [item[0] for item in batch]
        batch_z_s = [item[1] for item in batch]
        z_r, z_s = pack_sequence(batch_z_r, enforce_sorted=False), pack_sequence(batch_z_s, enforce_sorted=False)
        labels = torch.cat([item[2].reshape(1, 2) for item in batch], dim=0)
        return z_r, z_s, labels

    def train(self, train_path, val_path, emb_params):

        model = LSTM(emb_params, self.lstm_dim, self.num_lstm, self.bi_dir,
                                 self.output_dim, self.dropout)
        self.trainer = NetworkTrainer(model, self.tau_good, self.tau_bad, self.batch_size, 
                                      self.learning_rate, self.update_step_every, self.epochs)
        self.model = self.trainer.train(train_path, val_path, collate=self._collate_train)
        
    def _collate_test(self, batch):
        batch_z_r = [item[1] for item in batch]
        batch_z_s = [item[2] for item in batch]
        z_r, z_s = pack_sequence(batch_z_r, enforce_sorted=False), pack_sequence(batch_z_s, enforce_sorted=False)
        keys = [item[0] for item in batch]
        return keys, z_r, z_s
        
    def embed(self, path, embedder, print_progress):

        if self.trainer == None:
            raise RuntimeError("Model is not trained. Train model before embedding. ")

        return self.trainer.embed(path, print_progress, collate=self._collate_test)

class CNNModel:
    def __init__(self, embedding_size=100, output_size=100, kernels=[5], batch_size=64, dropout=0.1, 
                       epochs=30, learning_rate=0.001, tau_good=0.2, tau_bad=-0.2):
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
        return report.get_report_raw(), report.get_summary_raw()

    def _get_list_of_documents(self, element):
        return element

    def train(self, train_path, val_path, emb_params):

        model = CNN(emb_params, self.embedding_size, self.output_size, self.kernels, self.dropout)
        self.trainer = NetworkTrainer(model, self.tau_good, self.tau_bad, self.batch_size, 
                                      self.learning_rate, self.update_step_every, self.epochs)
        self.model = self.trainer.train(train_path, val_path)

    def embed(self, path, embedder, print_progress):

        if self.trainer == None:
            raise RuntimeError("Model is not trained. Train model before embedding. ")

        return self.trainer.embed(path, print_progress)
       

class WordLSTMModel:
    def __init__(self, embedding_dim=100, lstm_dim=100, num_lstm=1, output_dim=100, batch_size=64, dropout=0.1, epochs=30, learning_rate=0.001, update_step_every=10, tau_good=0.2, tau_bad=-0.2):
        self.name = 'WordLSTM'
        self.storage_format = ''
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.num_lstm = num_lstm
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.update_step_every = update_step_every
        self.tau_good = tau_good
        self.tau_bad = tau_bad
        self.trainer = None

    def _get_documents(self, report):
        return report.get_report_raw(), report.get_summary_raw()

    def _get_list_of_documents(self, element):
        return element

    def train(self, train_path, val_path, input_size):

        model = LSTM(input_size, self.lstm_dim, self.num_lstm, 
                                 self.output_dim, self.dropout, self.embedding_dim)
        self.trainer = NetworkTrainer(model, self.tau_good, self.tau_bad, self.batch_size, 
                                      self.learning_rate, self.update_step_every, self.epochs)
        self.model = self.trainer.train(train_path, val_path)

    def embed(self, path, embedder, print_progress):

        if self.trainer == None:
            raise RuntimeError("Model is not trained. Train model before embedding. ")

        return self.trainer.embed(path, print_progress)
        

class FFNModel:
    def __init__(self, layers=[100], batch_size=64, dropout=0.1, epochs=30, learning_rate=0.001, 
                       tau_good=0.2, tau_bad=-0.2):

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
        #self.loss = CosineLoss(tau_good, tau_bad, self.device)

    def _get_documents(self, report):
        return report.get_report_raw(), report.get_summary_raw()

    def _get_list_of_documents(self, element):
        doc1 = {'id': '%s_report' % element['__key__'], 'doc': element['report.pyd']}
        doc2 = {'id': '%s_summary' % element['__key__'], 'doc': element['summary.pyd']}
        return [doc1, doc2]

    def _training_step(self, trainLoader):

        trainLoss = 0
        trainAccuracy = 0
        length = 0
        self.model.train()
        for report_batch, summary_batch, label_batch in trainLoader:

            length += report_batch.shape[0]
            
            self.optimizer.zero_grad()
            z_r, z_s = self.model(report_batch.to(self.device), summary_batch.to(self.device))
            train_loss = cosine_loss(z_r, z_s, label_batch.to(self.device), self.tau_good, self.tau_bad, self.device)
            #train_loss = self.loss.loss(z_r, z_s,  label_batch.to(self.device))
            #train_loss = noise_aware_cosine_loss(z_r, z_s, label_batch.to(self.device), self.device)
            train_loss.backward()
            self.optimizer.step()
            trainLoss += train_loss.item()
        self.scheduler.step()
        return trainLoss/length

    def _validation_step(self, valloader):

        with torch.no_grad():
            self.model.eval()
            valLoss = 0
            valAccuracy = 0
            length = 0
            for report_batch, summary_batch, label_batch in valloader:
                length += report_batch.shape[0]

                report_batch = report_batch.to(self.device)
                summary_batch = summary_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                z_r, z_s = self.model(report_batch.to(self.device), summary_batch.to(self.device))
                val_loss = cosine_loss(z_r, z_s, label_batch.to(self.device), self.tau_good, self.tau_bad, self.device)
                #val_loss = self.loss.loss(z_r, z_s,  label_batch.to(self.device))
                #val_loss = noise_aware_cosine_loss(z_r, z_s, label_batch.to(self.device), self.device)
                #val_acc = pp.accuracy(predY, batchY)
                valLoss += val_loss.item()
                #valAccuracy += val_acc.item()
        return valLoss/length

    def _unpack_data_train(self, element):
        return (element['report.pth'], element['summary.pth'], element['labels.pth'])

    def train(self, train_path, val_path, emb_params):

        train = ReportData(train_path, shuffle_buffer_size=1000, apply=self._unpack_data_train, 
                                       batch_size=self.batch_size)
        if val_path != None:
            val = ReportData(val_path, shuffle_buffer_size=1000, apply=self._unpack_data_train, 
                                       batch_size=self.batch_size)
        else:
            val = None

        self.model = FFN(emb_params, self.layers, self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.update_step_every, gamma=0.1)

        print("\nTraining FFN model...")

        for e in range(1, self.epochs + 1):

            print("\nEpoch %s" % e)
            train_loss = self._training_step(train)
            if val != None:
                val_loss = self._validation_step(val)
                print('Train Loss = %.3f\tVal Loss = %.3f' % (train_loss, val_loss))                
            else:
                print('Train Loss = %.3f' % train_loss)

    def _unpack_data_test(self, element):
        return (element['__key__'], element['report.pth'], element['summary.pth'])

    def embed(self, path, embedder, print_progress):

        data = ReportData(path, apply=self._unpack_data_test, batch_size=self.batch_size, 
                                print_progress=print_progress)
        with torch.no_grad():
            iterator = iter(data)
            keys, z_r, z_s = [], [], []
            while True:
                if len(keys) == 0:
                    try:
                        keybatch, reportbatch, summarybatch = next(iterator)
                        new_z_r, new_z_s = self.model(reportbatch.to(self.device), summarybatch.to(self.device))
                        keys.extend(keybatch)
                        z_r.extend(new_z_r.to('cpu'))
                        z_s.extend(new_z_s.to('cpu'))
                    except StopIteration:
                        break
                yield {'id': keys.pop(0), 'z_r': z_r.pop(0), 'z_s': z_s.pop(0)}

    

class EmptyModel:
    def __init__(self):
        self.name = 'Baseline'
        self.storage_format = ''

    def _get_documents(self, report):
        return report.get_report_raw(), report.get_summary_raw()

    def _get_list_of_documents(self, element):
        doc1 = {'id': '%s_report' % element['__key__'], 'doc': element['report.pyd']}
        doc2 = {'id': '%s_summary' % element['__key__'], 'doc': element['summary.pyd']}
        return [doc1, doc2]

    def train(self, train, val, emb_params):
        pass

    def embed(self, path, embedder, print_progress):

        data = ReportData(path, apply=False, print_progress=print_progress)
        for element in data:
            yield {'id': element['__key__'], 
                   'z_r': embedder.embed(element['report.pyd']), 
                   'z_s': embedder.embed(element['summary.pyd'])}
    

class SummaryQualityModel:
    def __init__(self, embedder=None, model=EmptyModel()):
        
        self.embedder = embedder
        self.model = model
        self.storage_path = 'data/restructured/%s%s' % (model.storage_format, embedder.storage_format)
        self.embedded_path = 'models/%s/data/%s' % (model.name, embedder.unique_name)

    def _store_documents(self, element):

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

        out = {}
        out['__key__'] = element['__key__']

        if isinstance(element['report.pyd'][0], list):
            z_r = np.array([self.embedder.embed(doc) for doc in element['report.pyd']])
        else:
            z_r = self.embedder.embed(element['report.pyd'])
        out['report.pth'] = torch.from_numpy(z_r).float()

        if isinstance(element['summary.pyd'][0], list):
            z_s = np.array([self.embedder.embed(doc) for doc in element['summary.pyd']])
        else:
            z_s = self.embedder.embed(element['summary.pyd']) 
        out['summary.pth'] = torch.from_numpy(z_s).float()

        if 'labels.pyd' in element.keys():
            out['labels.pth'] = torch.from_numpy(np.asarray(element['labels.pyd'])).float()

        return out


    def _create_dataset(self, dataname, data, overwrite):

        data_path = '%s_%s' % (self.storage_path, dataname)
        ReportData(data_path).create(data=data, overwrite=overwrite, apply=self._store_documents)
        return data_path

    def _create_embedded_dataset(self, dataname, input_path, overwrite):

        if self.model.name == 'Baseline':
            return input_path
        input_data = ReportData(input_path, apply=False)
        data_path = '%s_%s' % (self.embedded_path, dataname)  
        ReportData(data_path).create(data=input_data, overwrite=overwrite, apply=self._store_embeddings)
        return data_path

    def _train_embedder(self, data_path, input_data, overwrite):
        data = ReportData(data_path, apply=self.model._get_list_of_documents)
        self.embedder.train(data, input_data, overwrite)

    def prepare_data(self, dataname, data, overwrite=False, overwrite_emb=False, train_embedder=False):

        if dataname == None:
            return None

        data_path = self._create_dataset(dataname, data, overwrite)
        if train_embedder:
            self._train_embedder(data_path, data, overwrite)
        embedded_data_path = self._create_embedded_dataset(dataname, data_path, overwrite_emb)
        return embedded_data_path

    def train(self, train_name, train_data, val_name=None, val_data=None, overwrite=False, overwrite_emb=False, train_embedder=True):

        train_path = self.prepare_data(train_name, train_data, overwrite=overwrite, overwrite_emb=overwrite_emb, train_embedder=train_embedder)
        val_path = self.prepare_data(val_name, val_data, overwrite=overwrite, overwrite_emb=overwrite_emb)
        self.model.train(train_path, val_path, emb_params=self.embedder.params)

    def embed(self, data_name, data, overwrite=False, print_progress=True, overwrite_emb=False):

        path = self.prepare_data(data_name, data, overwrite=overwrite, overwrite_emb=overwrite_emb)
        return self.model.embed(path, self.embedder, print_progress)

    def predict(self, data_name, data, overwrite=False, overwrite_emb=False):

        data = self.embed(data_name, data, overwrite=overwrite, overwrite_emb=overwrite_emb)
        keys = []
        qualities = []

        print('\nPredicting quality of %s' % data_name)
        for element in data:
            keys.append(element['id'])
            qualities.append(1 - spatial.distance.cosine(element['z_r'], element['z_s']))

        return pd.Series(qualities, index=keys)

    def save(self, modelname='default'):
        path = 'models/%s/%s.pkl' % (self.model.name, modelname)
        Path(path).mkdir(exist_ok=True, parents=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, modelname='default'):
        path = 'models/%s/%s.pkl' % (self.model.name, modelname)
        with open(path, 'rb') as f:
            self = pickle.load(f)
        return self




        