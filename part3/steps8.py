import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

def is_kaggle_gpu_enabled():
    """Return whether GPU is enabled in the running Kaggle kernel"""
    from tensorflow.python.client import device_lib
    return len(device_lib.list_local_devices()) > 2
print('GPU is enabled: ',is_kaggle_gpu_enabled())


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(163840, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)#initially was 10
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))#F.relu(self.conv1(x)), (2, 2))
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)#F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.flatten(x)
        return x

def train(model, train_loader, epochs, overfit_batch=False):
    #Use GPU
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4,weight_decay = 1e-5)

    if overfit_batch==True:
        batch = next(iter(train_loader)) #train with only one batch
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            gpu = next(model.parameters()).device
            inputs, labels, lengths = batch
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            optimizer.zero_grad()
            y_preds = model(inputs).double()

            loss = criterion(y_preds, labels.double())
            loss.backward()

            optimizer.step()

            train_loss = loss.data.item()

            if epoch%5 == 0:
                print("Epoch %d with loss: %f" %(epoch, train_loss))
    else:
        for epoch in range(epochs):
            model.train()
            for index, batch in enumerate(train_loader, 1):
                train_loss = 0.0

                gpu = next(model.parameters()).device

                inputs, labels, lengths = batch
                
                inputs = inputs.double()
                inputs = inputs.to(gpu)
                labels = labels.to(gpu)

                optimizer.zero_grad()
                y_preds = model(inputs).double()
#                 print(y_preds)
#                 print(labels)
                loss = criterion(y_preds, labels.double())
                loss.backward()

                optimizer.step()

                train_loss = loss.data.item()

            if epoch%3 == 0:
                print("Epoch %d with training loss: %f" %(epoch, train_loss))
            
#             if epoch%25 == 0:

#                 for index, batch in enumerate(val_loader, 1):
#                     val_loss = 0.0

#                     gpu = next(model.parameters()).device

#                     inputs, labels, lengths = batch

#                     inputs = inputs.to(gpu)
#                     labels = labels.to(gpu)

#                     #optimizer.zero_grad()
#                     y_preds = model(inputs).double()

#                     loss = criterion(y_preds, labels.double())

#                     #optimizer.step()

#                     val_loss = loss.data.item()

#                     #if epoch%100 == 0:
#                 print("Epoch %d with validation loss: %f" %(epoch, val_loss))

"""# Βήμα 8"""

import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
import re

# Combine similar classes and remove underrepresented classes
class_mapping = {
    'Rock': 'Rock',
    'Psych-Rock': 'Rock',
    'Indie-Rock': None,
    'Post-Rock': 'Rock',
    'Psych-Folk': 'Folk',
    'Folk': 'Folk',
    'Metal': 'Metal',
    'Punk': 'Metal',
    'Post-Punk': None,
    'Trip-Hop': 'Trip-Hop',
    'Pop': 'Pop',
    'Electronic': 'Electronic',
    'Hip-Hop': 'Hip-Hop',
    'Classical': 'Classical',
    'Blues': 'Blues',
    'Chiptune': 'Electronic',
    'Jazz': 'Jazz',
    'Soundtrack': None,
    'International': None,
    'Old-Time': None
}


def torch_train_val_split(
        dataset, batch_train, batch_eval,
        val_size=.2, shuffle=True, seed=None):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=batch_train,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset,
                            batch_size=batch_eval,
                            sampler=val_sampler)
    return train_loader, val_loader


def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return spectrogram.T


def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram.T


def read_spectrogram(spectrogram_file, chroma=True):
    # with open(spectrogram_file, "r") as f:
    spectrograms = np.load(spectrogram_file)
    # spectrograms contains a fused mel spectrogram and chromagram
    # Decompose as follows
    return spectrograms.T

    
def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return spectrogram.T


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[:self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1

        
class SpectrogramDataset_multi(Dataset):
    def __init__(self, path,emotion, class_mapping=None, train=True, max_length=-1, read_spec_fn=read_spectrogram):
        t = 'train' if train else 'test'
        if(emotion=='valence'):
            self.emotion = 1
        else:
            if(emotion=='energy'):
                self.emotion = 2
            else:
                if(emotion=='danceability'):
                    self.emotion = 3
        p = os.path.join(path, t)
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('double')#self.label_transformer.fit_transform(labels)).astype('double')#(initially int64)

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            l = l[0].split(",")
            label = l[self.emotion]
#             print(label)
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0].split('.')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        # TODO: Inspect output and comment on how the output is formatted
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self):
        return len(self.labels)

#valence cnn train
multitask_dataset = SpectrogramDataset_multi("../input/patreco3-multitask-affective-music/data/multitask_dataset/",'valence', train=True)
train_loader_multi_valence, test_loader_multi_valence = torch_train_val_split(multitask_dataset, 32 ,32, val_size=.33)
net_valence = Net().double()
train(net_valence, train_loader_multi_valence, 2, overfit_batch=False)
torch.save(net_valence, './cnn_valence.pt')

#valence cnn energy
multitask_dataset = SpectrogramDataset_multi("../input/patreco3-multitask-affective-music/data/multitask_dataset/",'energy', train=True)
train_loader_multi_energy, test_loader_multi_energy = torch_train_val_split(multitask_dataset, 32 ,32, val_size=.33)
net_energy = Net().double()
train(net_energy, train_loader_multi_energy, 25, overfit_batch=False)
torch.save(net_energy, './cnn_energy.pt')

#valence cnn danceability
multitask_dataset = SpectrogramDataset_multi("../input/patreco3-multitask-affective-music/data/multitask_dataset/",'danceability', train=True)
train_loader_multi_danceability, test_loader_multi_danceability = torch_train_val_split(multitask_dataset, 32 ,32, val_size=.33)
net_danceability = Net().double()
train(net_danceability, train_loader_multi_danceability, 25, overfit_batch=False)
torch.save(net_danceability, './cnn_danceability.pt')


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths =  [feat.shape[0] for feat in feats]# Find the lengths 

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = []
        # --------------- Insert your code here ---------------- #
        max_length = max(self.lengths)
        padded = np.array([np.pad(example, ((0, max_length-len(example)), (0, 0)), mode='constant') for example in x])
        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False,dropout=0):
        super(BasicLSTM, self).__init__()
        self.input_dim = input_dim
        self.rnn_size = rnn_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.rnn = torch.nn.LSTM(input_size = input_dim, hidden_size = rnn_size, num_layers = num_layers, batch_first = True, bidirectional=self.bidirectional,dropout = dropout).double() #,dtype=torch.double
        self.linear = torch.nn.Linear(self.feature_size, output_dim).double()

    def forward(self, x, lengths): 
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index
            lengths: N x 1
         """
        GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_dim = self.rnn_size
        if(self.bidirectional):
            ht_num_layers = self.num_layers*2 # for bidirectional we need double size of layers
        else:
            ht_num_layers = self.num_layers
        h_t = torch.zeros(ht_num_layers, x.shape[0], hidden_dim).double().to(GPU)
        c_t = torch.zeros(ht_num_layers, x.shape[0], hidden_dim).double().to(GPU)

        out, h_n = self.rnn(x, (h_t,c_t)) #forward pass for LSTM
        last_outputs = self.linear(self.last_timestep(out, lengths, self.bidirectional)).double()
        last_outputs = torch.flatten(last_outputs)

        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1).to(GPU)
        return outputs.gather(1, idx).squeeze()

def train_lstm(model, train_loader, epochs, overfit_batch=False):
    #Use GPU
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4,weight_decay = 1e-5)

    if overfit_batch==True:
        batch = next(iter(train_loader)) #train with only one batch
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            gpu = next(model.parameters()).device
            inputs, labels, lengths = batch
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            optimizer.zero_grad()
            y_preds = model(inputs, lengths).double()

            loss = criterion(y_preds, labels.double())
            loss.backward()

            optimizer.step()

            train_loss = loss.data.item()

            if epoch%5 == 0:
                print("Epoch %d with loss: %f" %(epoch, train_loss))
    else:
        for epoch in range(epochs):
            model.train()
            for index, batch in enumerate(train_loader, 1):
                train_loss = 0.0

                gpu = next(model.parameters()).device

                inputs, labels, lengths = batch
                
                inputs = inputs.double()
                inputs = inputs.to(gpu)
                labels = labels.to(gpu)

                optimizer.zero_grad()
                y_preds = model(inputs, lengths).double()
#                 print(y_preds)
#                 print(labels)
                loss = criterion(y_preds, labels.double())
                loss.backward()

                optimizer.step()

                train_loss = loss.data.item()

            if epoch%3 == 0:
                print("Epoch %d with training loss: %f" %(epoch, train_loss))
            
#             if epoch%25 == 0:

#                 for index, batch in enumerate(val_loader, 1):
#                     val_loss = 0.0

#                     gpu = next(model.parameters()).device

#                     inputs, labels, lengths = batch

#                     inputs = inputs.to(gpu)
#                     labels = labels.to(gpu)

#                     #optimizer.zero_grad()
#                     y_preds = model(inputs, lengths).double()

#                     loss = criterion(y_preds, labels.double())

#                     #optimizer.step()

#                     val_loss = loss.data.item()

#                     #if epoch%100 == 0:
#                 print("Epoch %d with validation loss: %f" %(epoch, val_loss))

lstm_multi_valence = BasicLSTM(input_dim = multitask_dataset.feat_dim, rnn_size = 32, output_dim = 1, num_layers = 4, bidirectional=True).double()
lstm = train_lstm(lstm_multi_valence, train_loader_multi_valence, 2, overfit_batch=False)
torch.save(lstm_multi_valence, './lstm_valence.pt')

lstm_multi_energy = BasicLSTM(input_dim = multitask_dataset.feat_dim, rnn_size = 32, output_dim = 1, num_layers = 4, bidirectional=True).double()
train_lstm(lstm_multi_energy, train_loader_multi_energy, 20, overfit_batch=False)
torch.save(lstm_multi_energy, './lstm_energy')

lstm_multi_danceability = BasicLSTM(input_dim = multitask_dataset.feat_dim, rnn_size = 32, output_dim = 1, num_layers = 4, bidirectional=True).double()
train_lstm(lstm_multi_danceability, train_loader_multi_danceability, 20, overfit_batch=False)#test_loader_multi_danceability
torch.save(lstm_multi_danceability, './lstm_danceability')

from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def evaluate(model, test_loader):
    GPU = torch.device("cpu")
    model.to(GPU)

    #criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0

    correlations = []

    gpu = next(model.parameters()).device
    print(gpu)
    with torch.no_grad():
        for index, batch in enumerate(test_loader, 1):
            inputs, labels, lengths = batch

            inputs = inputs.to(gpu)
            labels = labels.to(gpu)
            lengths = lengths.to(gpu)

            y_preds = model(inputs, lengths)  # EX9
            spear = spearmanr(y_preds,labels).to(gpu)
            correlations.append(spear)
            #print(y_preds)
            #loss = criterion(y_preds, labels)

            #prediction
            #y_preds_arg = torch.argmax(y_preds, dim=1)

            #y_pred.append(y_preds_arg.cpu().numpy())
            #y.append(labels.cpu().numpy())

            #test_loss += loss.data.item()

    return sum(correlations)/len(correlations)

energy = evaluate(net_energy,test_loader_multi_energy)
valence = evaluate(net_valence,test_loader_multi_valence)
danceability = evaluate(net_danceability,test_loader_multi_danceability)
mean = (energy+valence+danceability)/3