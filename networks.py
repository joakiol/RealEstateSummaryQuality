import torch.nn as nn
import torch.nn.functional as F
import torch
from data import ReportData
import numpy as np

class NetworkTrainer:
    def __init__(self, model, tau_good, tau_bad, batch_size, lr, update_step_every, epochs):
        """Trainer class for training pytorch networks, as well as predicting qualities. 

        Args:
            model (network object (FFN/LSTM/CNN)): Model to train. 
            tau_good (float): Which tau_good to use for training. 
            tau_bad (float): Which tau_bad to use for training. 
            batch_size (int): Batch size for training
            lr (float): Initial learning rate for training. 
            update_step_every (int): How many epochs to perform before updating learning rate. 
            epochs (int): Number of training epochs. 
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tau_good = tau_good
        self.tau_bad = tau_bad
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, update_step_every, gamma=0.1)

    def _training_step(self, train_loader):
        """One training epoch for neural network. """
        trainLoss = 0
        trainAccuracy = 0
        length = 0
        self.model.train()
        for report_batch, summary_batch, label_batch in train_loader:
            length += label_batch.shape[0]
            self.optimizer.zero_grad()
            z_r, z_s = self.model(report_batch.to(self.device), summary_batch.to(self.device))
            train_loss = cosine_loss(z_r, z_s, label_batch.to(self.device), self.tau_good, self.tau_bad, self.device)
            train_loss.backward()
            self.optimizer.step()
            trainLoss += train_loss.item()
        self.scheduler.step()
        return trainLoss/length

    def _validation_step(self, val_loader):
        """One validation epoch for neural network."""
        with torch.no_grad():
            self.model.eval()
            valLoss = 0
            valAccuracy = 0
            length = 0
            for report_batch, summary_batch, label_batch in val_loader:
                length += label_batch.shape[0]
                z_r, z_s = self.model(report_batch.to(self.device), summary_batch.to(self.device))
                val_loss = cosine_loss(z_r, z_s, label_batch.to(self.device), self.tau_good, self.tau_bad, self.device)
                valLoss += val_loss.item()
        return valLoss/length

    def _unpack_data_train(self, element):
        """This method is used by train method, to determine how training data should
        be unpacked from webdataset format. """
        return (element['report.pth'], element['summary.pth'], element['labels.pth'])

    def train(self, train_path, val_path, collate=None):
        """Train neural network

        Args:
            train_path (str): Path to training data. 
            val_path (str): Path to validation data. 
            collate (func, optional): Collate func is used with LSTM, for packed sequence stuff. 
                                      Defaults to None.

        Returns:
            network object (FFN/LSTM/CNN): Trained neural network. 
        """        
        train = ReportData(train_path, shuffle_buffer_size=1000, apply=self._unpack_data_train, 
                                        batch_size=self.batch_size, collate=collate)
        if val_path != None:
            val = ReportData(val_path, shuffle_buffer_size=1000, apply=self._unpack_data_train, 
                                        batch_size=self.batch_size, collate=collate)
        else:
            val = None

        print("\nTraining model...")

        for e in range(1, self.epochs + 1):

            print("\nEpoch %s" % e)
            train_loss = self._training_step(train)
            if val != None:
                val_loss = self._validation_step(val)
                print('Train Loss = %.3f\tVal Loss = %.3f' % (train_loss, val_loss))                
            else:
                print('Train Loss = %.3f' % train_loss)

        return self.model

    def _unpack_data_test(self, element):
        """This method is used by embed-method, to determine how test data should
        be unpacked from webdataset format. """
        return (element['__key__'], element['report.pth'], element['summary.pth'])

    def embed(self, path, print_progress, collate=None):
        """Generator for embedding data at input path in a memory-friendly fashion. 
        Creates reports and summaries in the conceptual summary content space, where
        summary quality can be measured by cosine similarity. 

        Args:
            path (str): Path to data to embed. 
            print_progress (boolean): Whether to print progress of embedding. 
            collate (func, optional): Collate func is used with LSTM, for packed sequence stuff. 
                                      Defaults to None.

        Yields:
            dict{'id', 'z_r', 'z_s'}: Dictionary with id, embedder report (z_r) and embedded 
                                      summary (z_s). 
        """
        data = ReportData(path, apply=self._unpack_data_test, batch_size=self.batch_size, 
                                print_progress=print_progress, collate=collate)
        with torch.no_grad():
            self.model.eval()
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


class FFN(nn.Module):
    def __init__(self, input_nodes, layers, dropout):
        """FFN network class. See master thesis for architecture details. 

        Args:
            input_nodes (int): Dimensionality of input embeddings. 
            layers (list[int]): Number of nodes to use in each layer. Number of layers 
                                becomes length of layers list. 
            dropout (float): Dropout rate. 
        """        
        super(FFN, self).__init__()
        self.layers = [nn.Linear(input_nodes, layers[0])]
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.layers = nn.ModuleList(self.layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(layers[i]) for i in range(len(layers) - 1)])

    def forward(self, r, s):
        """Applies network on batch of reports and summaries.

        Args:
            r (tensor): Report batch, already embedded by LSA/Doc2vec
            s (tensor): Summary batch, already embedded by LSA/Doc2vec

        Returns:
            z_r (tensor): Output embeddings of report batch, ready for measuring quality (cossim).
            z_s (tensor): Output embeddings of summary batch, ready for measuring quality (cossim).
        """        
        for i in range(len(self.layers) - 1):
            r = self.layers[i](r)
            r = self.relu(r)
            r = self.batchnorm[i](r)
            r = self.dropout(r)

            s = self.layers[i](s)
            s = self.relu(s)
            s = self.batchnorm[i](s)
            s = self.dropout(s)

        z_r = self.layers[-1](r)
        z_s = self.layers[-1](s)
        return z_r, z_s


class LSTM(nn.Module):
    def __init__(self, input_nodes, lstm_dim, num_lstm, bi_dir, output_dim, dropout):
        """LSTM network class. See master thesis for architecture details. 

        Args:
            input_nodes (int): Dimensionality of input embeddings. 
            lstm_dim (int): Dimensionality of LSTM cell. 
            num_lstm (int): Number of LSTM layers. 
            bi_dir (boolean): Whether bi-directional LSTM layers are employed or not. 
            output_dim (int): Output dimensionality of final, fully connected linear layer. 
            dropout (float): Dropout rate. 
        """        
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_nodes, lstm_dim, num_layers=num_lstm, dropout=dropout, bidirectional=bi_dir)
        self.output_layer = nn.Linear(lstm_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, r, s):
        """Applies network on batch of reports and summaries.

        Args:
            r (tensor): Report batch, already embedded by LSA/Doc2vec
            s (tensor): Summary batch, already embedded by LSA/Doc2vec

        Returns:
            z_r (tensor): Output embeddings of report batch, ready for measuring quality(cossim).
            z_s (tensor): Output embeddings of summary batch, ready for measuring quality(cossim).
        """     
        lstm_out_r, (ht_r, ct_r) = self.lstm(r)
        lstm_out_s, (ht_s, ct_s) = self.lstm(s)
        z_r_out = self.dropout(ht_r[-1])
        z_s_out = self.dropout(ht_s[-1])
        return self.output_layer(z_r_out), self.output_layer(z_s_out)



class CNN(nn.Module):
    def __init__(self, params, embedding_size, output_size, kernels, dropout):
        """CNN network class. See master thesis for architecture details. 

        Args:
            params (dict{vocab_length/emb_matrix}): Parameters from embedder, which holds
                                                    necessary information. 
            embedding_size (int): Dimensionality of word embeddings in EmbLayer/Word2vec. 
            output_size (int): Number of filters per filter size and nodes in final linear layer.
            kernels (list[int]): List of filter sizes. Total number of filters becomes
                                 len(kernels)*output_size. 
            dropout (float): Dropout rate. 
        """        
        super(CNN, self).__init__()
        if 'vocab_length' in params.keys():
            vocab_length = params['vocab_length']
            self.embedding = nn.Embedding(vocab_length+1, embedding_size, padding_idx=0)
        elif 'emb_matrix' in params.keys():
            wordvectors = params['emb_matrix'].vectors
            zeros = np.zeros((1, len(wordvectors[0])))
            wordvectors = np.concatenate((zeros, wordvectors), axis=0)
            self.embedding = nn.Embedding(len(wordvectors), len(wordvectors[0]), padding_idx=0)
            self.embedding.weight = nn.Parameter(torch.from_numpy(wordvectors).float())
            self.embedding.weight.requires_grad = False

        self.layers = nn.ModuleList(
            [nn.Conv2d(1, output_size, [kernel_size, embedding_size], padding=(kernel_size-1, 0)) 
            for kernel_size in kernels]
        )
        self.linear = nn.Linear(output_size*len(kernels), output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, r, s):
        """Applies network on batch of reports and summaries.

        Args:
            r (tensor): Report batch, already embedded by VocabularyEmbedder/Word2vec.
            s (tensor): Summary batch, already embedded by VocabularyEmbedder/Word2vec.

        Returns:
            z_r (tensor): Output embeddings of report batch, ready for measuring quality(cossim).
            z_s (tensor): Output embeddings of summary batch, ready for measuring quality(cossim).

        """
        r, s = self.embedding(r.long()), self.embedding(s.long())
        r, s = torch.unsqueeze(r, 1), torch.unsqueeze(s, 1)
        r_list, s_list = [], []
        for i, conv in enumerate(self.layers):
            new_r, new_s = conv(r), conv(s)
            new_r, new_s = self.relu(new_r), self.relu(new_s)
            new_r, new_s = torch.squeeze(new_r, -1), torch.squeeze(new_s, -1)
            new_r, new_s = F.max_pool1d(new_r, new_r.size(2)), F.max_pool1d(new_s, new_s.size(2))
            r_list.append(new_r)
            s_list.append(new_s)

        r, s = torch.cat(r_list, 2), torch.cat(s_list, 2)
        r, s = r.view(r.size(0), -1), s.view(s.size(0), -1)
        r, s = self.dropout(r), self.dropout(s)
        return self.linear(r), self.linear(s)


def cosine_loss(z_r, z_s, labels, tau_good, tau_bad, device):
    """Noise aware cosine embedding loss, to use for training networks. 

    Args:
        z_r (tensor): Report batch embedded by neural network. 
        z_s (tensor): Summary batch embedded by neural network. 
        labels (tensor): Labels for batch. 
        tau_good (float): Which tau_good to use in loss function. 
        tau_bad (float): Which tau_bad to use in loss function. 
        device (str): Device for calculating loss, in case of cuda. 

    Returns:
        float: Sum of loss for reports/summaries in the batch. 
    """    
    z_r = nn.functional.normalize(z_r, p=2, dim=1)
    z_s = nn.functional.normalize(z_s, p=2, dim=1)
    cos_sim = torch.sum(z_r*z_s, dim=1)
    ones = torch.ones(z_r.shape[0], dtype=torch.int32, device=device)
    good = torch.clamp(tau_good * ones - cos_sim, min=0)
    bad = torch.clamp(cos_sim - tau_bad * ones, min=0)
    return torch.sum(labels[:,0] * bad + labels[:,1] * good)