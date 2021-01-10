import torch.nn as nn
import torch.nn.functional as F
import torch
from data import ReportData
import numpy as np

class NetworkTrainer:
    def __init__(self, model, tau_good, tau_bad, batch_size, lr, update_step_every, epochs):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tau_good = tau_good
        self.tau_bad = tau_bad
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, update_step_every, gamma=0.1)

    def _training_step(self, train_loader):

        trainLoss = 0
        trainAccuracy = 0
        length = 0
        self.model.train()
        for report_batch, summary_batch, label_batch in train_loader:

            length += label_batch.shape[0]
            
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

    def _validation_step(self, val_loader):

        with torch.no_grad():
            self.model.eval()
            valLoss = 0
            valAccuracy = 0
            length = 0
            for report_batch, summary_batch, label_batch in val_loader:
                length += label_batch.shape[0]
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

    def train(self, train_path, val_path, collate=None):

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
        return (element['__key__'], element['report.pth'], element['summary.pth'])

    def embed(self, path, print_progress, collate=None):

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



class LSTM(nn.Module):
    def __init__(self, input_nodes, lstm_dim, num_lstm, bi_dir, output_dim, dropout, embedding=None):
        super(LSTM, self).__init__()

        if embedding != None:
            self.embedding = nn.Embedding(input_nodes+1, embedding, padding_idx=0)
            self.lstm = nn.LSTM(embedding, lstm_dim, num_layers=num_lstm, dropout=dropout, batch_first=True, bidirectional=bi_dir)
        else:
            self.embedding = None
            self.lstm = nn.LSTM(input_nodes, lstm_dim, num_layers=num_lstm, dropout=dropout, bidirectional=bi_dir)

        self.output_layer = nn.Linear(lstm_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, z_r, z_s):

        if self.embedding != None:
            z_r, z_s = self.embedding(z_r.long()), self.embedding(z_s.long())

        lstm_out_r, (ht_r, ct_r) = self.lstm(z_r)
        lstm_out_s, (ht_s, ct_s) = self.lstm(z_s)
        z_r_out = self.dropout(ht_r[-1])
        z_s_out = self.dropout(ht_s[-1])
        return self.output_layer(z_r_out), self.output_layer(z_s_out)

class Attn(nn.Module):
    def __init__(self, input_nodes, attn_dim, num_attn, output_dim, dropout):
        super(Attn, self).__init__()

        self.key = nn.Linear(input_nodes, attn_dim)
        self.query = nn.Linear(input_nodes, attn_dim)
        self.value = nn.Linear(input_nodes, attn_dim)

        self.attn = nn.MultiheadAttention(input_nodes, num_attn, dropout=dropout)
        self.lstm = nn.LSTM(input_nodes, attn_dim, num_layers=num_attn, dropout=dropout)

        self.output_layer = nn.Linear(attn_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, z_r, z_s):

        z_r = nn.utils.rnn.pad_packed_sequence(z_r)
        z_s = nn.utils.rnn.pad_packed_sequence(z_s)

        Q = self.query(z_r[0])
        K = self.key(z_s[0])
        V = self.value(z_s[0])

        attn_output, _ = self.attn(Q, K, V)

        lstm_out_r, (ht_r, ct_r) = self.lstm(z_r[0])
        lstm_out_s, (ht_s, ct_s) = self.lstm(attn_output)
        z_r_out = self.dropout(ht_r[-1])
        z_s_out = self.dropout(ht_s[-1])
        return self.output_layer(z_r_out), self.output_layer(z_s_out)

       
        #return 


class FFN(nn.Module):
    def __init__(self, input_nodes, layers, dropout):
        super(FFN, self).__init__()


        self.layers = [nn.Linear(input_nodes, layers[0])]
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.layers = nn.ModuleList(self.layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(layers[i]) for i in range(len(layers) - 1)])

    def forward(self, r, s):
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

class CNN(nn.Module):
    def __init__(self, params, embedding_size, output_size, kernels, dropout):
        super(CNN, self).__init__()

        # Define embedding layers
        if 'vocab_length' in params.keys():
            vocab_length = params['vocab_length']
            self.embedding = nn.Embedding(vocab_length+1, embedding_size, padding_idx=0)
        elif 'emb_matrix' in params.keys():
            wordvectors = params['emb_matrix'].vectors
            zeros = np.zeros((1, len(wordvectors[0])))
            wordvectors = np.concatenate((zeros, wordvectors), axis=0)
            
            print(len(wordvectors), len(wordvectors[0]))
            self.embedding = nn.Embedding(len(wordvectors), len(wordvectors[0]), padding_idx=0)
            self.embedding.weight = nn.Parameter(torch.from_numpy(wordvectors).float())
            self.embedding.weight.requires_grad = False

        # Define convolution layers
        self.layers = nn.ModuleList(
            [nn.Conv2d(1, output_size, [kernel_size, embedding_size], padding=(kernel_size-1, 0)) 
            for kernel_size in kernels]
        )
        
        # Define max pooling layers
        #self.pool = nn.ModuleList([nn.MaxPool1d(kernel_size) for kernel_size in kernels])
       
        # Define fully connected linear layer
        self.linear = nn.Linear(output_size*len(kernels), output_size)

        #  Define activation layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        #self.batchnorm = nn.ModuleList([nn.BatchNorm1d(layers[i]) for i in range(len(layers) - 1)])

    def forward(self, r, s):

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




# class NoiseAwareLoss(nn.CosineSimilarity):
#     def __init__(self, device,):
#         super(NoiseAwareLoss, self).__init__()
#         self.device = device

#     def loss(z_r, z_s, labels):
#         cos_sim = self(z_r, z_s)
#         return torch.sum(labels[:,0]*torch.clamp())

def cosine_loss(z_r, z_s, labels, tau_good, tau_bad, device):
    z_r = nn.functional.normalize(z_r, p=2, dim=1)
    z_s = nn.functional.normalize(z_s, p=2, dim=1)
    cos_sim = torch.sum(z_r*z_s, dim=1)
    ones = torch.ones(z_r.shape[0], dtype=torch.int32, device=device)
    good = torch.clamp(tau_good * ones - cos_sim, min=0)
    bad = torch.clamp(cos_sim - tau_bad * ones, min=0)
    return torch.sum(labels[:,0] * bad + labels[:,1] * good)

# class CosineLoss:
#     def __init__(self, tau_good, tau_bad, device):
#         self.tau_good = tau_good
#         self.tau_bad = tau_bad
#         self.cosine = nn.CosineSimilarity()
#         self.device = device

#     def loss(self, z_r, z_s, labels):

#         length = z_r.shape[0]
#         ones = torch.ones(length, dtype=torch.int32, device=self.device)
#         cos_sim = self.cosine(z_r, z_s)

#         good = torch.clamp(self.tau_good * ones - cos_sim, min=0)
#         bad = torch.clamp(cos_sim - self.tau_bad * ones, min=0)
#         return torch.sum(labels[:,0] * bad + labels[:,1] * good)


# def noise_aware_cosine_loss(z_r, z_s, labels, device):

#     #cos_sim = nn.CosineSimilarity()

#     good = torch.ones([z_r.shape[0], 1], dtype=torch.int32, device=device)
#     bad = -torch.ones([z_r.shape[0], 1], dtype=torch.int32, device=device)

#     loss_good = F.cosine_embedding_loss(z_r, z_s, good, reduction='none', margin=-1)
#     loss_bad = F.cosine_embedding_loss(z_r, z_s, bad, reduction='none', margin=-1)
#     return torch.sum((labels[:,0] * loss_bad[0]) + labels[:,1] * (torch.clamp(loss_good[0], min=0) - 0*torch.ones(len(loss_good[0]), dtype=torch.int32, device=device)) )
    

# class NoiseAwareCosineLoss(nn.Module):
#     def __init__(self, batch_size):
#         super(NoiseAwareCosineLoss, self).__init__()
#         self.batch_size = batch_size

#     def forward(self, input1, input2, target):
#         loss_good = F.cosine_embedding_loss(input1, input2, torch.ones([self.batch_size, 1], dtype=torch.int32), reduction='none')
#         loss_bad = F.cosine_embedding_loss(input1, input2, torch.zeros([self.batch_size, 1], dtype=torch.int32), reduction='none')
#         return torch.dot(target[:,0].float(), loss_bad[0]) + torch.dot(target[:,1].float(), loss_good[0])
