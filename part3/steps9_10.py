import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
import re
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

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

        
class SpectrogramDataset(Dataset):
    def __init__(self, path, class_mapping=None, train=True, max_length=-1, read_spec_fn=read_spectrogram):
        t = 'train' if train else 'test'
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
            self.labels = np.array(self.label_transformer.fit_transform(labels)).astype('int64')

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = l[1]
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

def train(model, train_loader, val_loader, epochs, overfit_batch=False):
    #Use GPU
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = torch.nn.CrossEntropyLoss()
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
            y_preds = model(inputs)
            
            loss = criterion(y_preds, labels)
            loss.backward()

            optimizer.step()

            train_loss = loss.data.item()

            if epoch%3 == 0:
                print("Epoch %d with loss: %f" %(epoch, train_loss))
    else:
        for epoch in range(epochs):
            model.train()
            for index, batch in enumerate(train_loader, 1):
                train_loss = 0.0

                gpu = next(model.parameters()).device

                inputs, labels, lengths = batch
                
                inputs = inputs.to(gpu)
                labels = labels.to(gpu)

                optimizer.zero_grad()
                y_preds = model(inputs)
                
#                 y_preds = y_preds.cpu().detach().numpy()
#                 labels = labels.cpu().detach().numpy()
#                 print(y_preds.shape, labels.shape)

                loss = criterion(y_preds, labels)
                loss.backward()

                optimizer.step()

                train_loss = loss.data.item()

            if epoch%3 == 0:
                print("Epoch %d with training loss: %f" %(epoch, train_loss))
            
            if epoch%6 == 0:
                for index, batch in enumerate(val_loader, 1):
                    val_loss = 0.0

                    gpu = next(model.parameters()).device

                    inputs, labels, lengths = batch

                    inputs = inputs.to(gpu)
                    labels = labels.to(gpu)

                    y_preds = model(inputs)

                    loss = criterion(y_preds, labels)

                    val_loss = loss.data.item()

                print("Epoch %d with validation loss: %f" %(epoch, val_loss))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(148480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#mel
mel_specs = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_mel, val_loader_mel = torch_train_val_split(mel_specs, 32 ,32, val_size=.33)
test_mel = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
test_loader_mel = DataLoader(test_mel, batch_size=32)

# cnn = Net().double()
# train(cnn, train_loader_mel, val_loader_mel, 25)

# torch.save(cnn, './nn_9_mel')

cnn_9_mel = torch.load('/kaggle/input/cnn-9-mel/nn_9_mel')
print(cnn_9_mel)

# Freeze all parameters
for param in cnn_9_mel.parameters():
    param.requires_grad = False
    
# Change last layer to regression
cnn_9_mel.fc3 = nn.Linear(84, 1)
print(cnn_9_mel)

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

def train_regression(model, train_loader, epochs):
    #Use GPU
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = torch.nn.MSELoss().to(GPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4,weight_decay = 1e-5)
    
    losses=[]
    for epoch in range(epochs):
        model.train()
        for index, batch in enumerate(train_loader, 1):
            train_loss = 0.0

            gpu = next(model.parameters()).device

            inputs, labels, lengths = batch
            #  print(inputs[0].size())
            #  Its (1293,140) not (1293,128) which cnn was trained on
            #  so we crop the input

            inputs = inputs[:,:,:128]

            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            optimizer.zero_grad()
            y_preds = model(inputs)

            loss = criterion(y_preds, labels)
            loss.backward()

            optimizer.step()

        train_loss = loss.data.item()
        losses.append(train_loss)
        print("Epoch %d with training loss: %f" %(epoch, train_loss))
    return losses

#delete previous dataset so kaggle does not crash from memory
del mel_specs, train_loader_mel, val_loader_mel, test_loader_mel, test_mel

#valence cnn energy
multitask_dataset_valence = SpectrogramDataset_multi("../input/patreco3-multitask-affective-music/data/multitask_dataset/",'valence', train=True)
train_loader_multi_valence, test_loader_multi_valence = torch_train_val_split(multitask_dataset_valence, 32 ,32, val_size=.33)

cnn_9_mel.double()
train_losses = train_regression(cnn_9_mel, train_loader_multi_valence, 15)


plt.plot(train_losses)
plt.title('Transfered CNN loss on valence')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()

torch.save(cnn_9_mel, './cnn_9_transfered_valence')

from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def evaluate(model, test_loader):
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = torch.nn.MSELoss().to(GPU)
    model.eval()
    test_loss = 0.0

    y_pred = []
    y = []

    gpu = next(model.parameters()).device

    with torch.no_grad():
        for index, batch in enumerate(test_loader, 1):
            inputs, labels, lengths = batch
            inputs = inputs[:,:,:128]
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

            y_pred.append(y_preds.cpu().numpy())
            y.append(labels.cpu().numpy())
            
            test_loss += loss.data.item()

    return y, y_pred


y, y_pred = evaluate(cnn_9_mel, test_loader_multi_valence)
gt = np.array([])
for b in y:
    gt = np.append(gt,b)

pr = np.array([])
for b in y_pred:
    pr = np.append(pr,b)
    
from scipy.stats import spearmanr
print("Transfer learning spearman correlation: %f" %spearmanr(gt, pr)[0])


#STEP 10
class multitask_loss_function(nn.Module):
    
    def __init__(self):
        super(multitask_loss_function,self).__init__()
    
    def forward(self, logits, targets):
        #sizes are (32,3)
        GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = logits.to(GPU)
        targets = targets.to(GPU)

        loss1 = torch.nn.MSELoss().to(GPU)
        loss2 = torch.nn.MSELoss().to(GPU)
        loss3 = torch.nn.MSELoss().to(GPU)
        
#         print(logits[:,0].shape)
#         print(targets[:,0].shape)
#       # sizes are (32,) and given to nn.MSELoss()
        v_loss = loss1(logits[:,0],targets[:,0])
        a_loss = loss2(logits[:,1],targets[:,1])
        d_loss = loss3(logits[:,2],targets[:,2])
        
        loss = v_loss + a_loss + d_loss
        return loss

del multitask_dataset_valence, train_loader_multi_valence, test_loader_multi_valence

class MultitaskDataset(Dataset):
    def __init__(self, path, max_length=-1, read_spec_fn=read_fused_spectrogram, label_type='energy'):
        p = os.path.join(path, 'train')
        self.label_type = label_type
        self.index = os.path.join(path, "train_labels.txt")
        self.files, labels = self.get_files_labels(self.index)
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length) 
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels)

    def get_files_labels(self, txt):
        with open(txt, 'r') as fd:
            lines = [l.split(',') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            if self.label_type == 'valence':
                labels.append(float(l[1]))
            elif self.label_type == 'energy':
                labels.append(float(l[2]))
            elif self.label_type == 'danceability':
                labels.append(float(l[3].strip("\n")))
            else:
                labels.append([float(l[1]), float(l[2]), float(l[3].strip("\n"))])
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
        return files, labels
    

    def __getitem__(self, item):
        # Return a tuple in the form (padded_feats, valence, energy, danceability, length)
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self):
        return len(self.labels)

specs_multi = MultitaskDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         max_length=1293,
         label_type=-1,
         read_spec_fn=read_mel_spectrogram)

train_loader_multi, test_loader_multi = torch_train_val_split(specs_multi, 32 ,32, val_size=.33)



class Multitask_CNN(nn.Module):

    def __init__(self):
        super(Multitask_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(148480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_valence = nn.Linear(84, 1)
        self.fc_arousal = nn.Linear(84, 1)
        self.fc_danceability = nn.Linear(84, 1)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        valence = self.fc_valence(x)
        arousal = self.fc_arousal(x)
        danceability = self.fc_danceability(x)
        return valence, arousal, danceability
        
def train_regression_multitask(model, train_loader, epochs):
    #Use GPU
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = multitask_loss_function().to(GPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4,weight_decay = 1e-5)
    losses = []
    for epoch in range(epochs):
        model.train()
        for index, batch in enumerate(train_loader, 1):
            train_loss = 0.0

            gpu = next(model.parameters()).device

            inputs, labels, lengths = batch
            #  print(inputs[0].size())
            #  Its (1293,140) not (1293,128) which cnn was trained on
            #  so we crop the input

            inputs = inputs[:,:,:128]

            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            optimizer.zero_grad()
            valence, arousal, danceability = model(inputs)
    
            y_preds = [valence, arousal, danceability]
            y_preds = torch.cat(y_preds, dim=1)
    
#             print('y_preds shape:', y_preds.size()) #(32,3)
            #labels is (32,3)
            loss = criterion(y_preds, labels)
        
            loss.backward()

            optimizer.step()

        train_loss = loss.data.item()
        losses.append(train_loss)

        print("Epoch %d with training loss: %f" %(epoch, train_loss))
    return losses

cnn_multi = Multitask_CNN().double()
train_losses = train_regression_multitask(cnn_multi, train_loader_multi, 25)


print(cnn_multi)
torch.save(cnn_multi, './cnn_10_multi')



plt.plot(train_losses)
plt.title('Multitask CNN loss')
plt.xlabel('Epochs')
plt.ylabel('MSE sum on VAD')
plt.show()

def evaluate_multitask(model, test_loader):
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = multitask_loss_function().to(GPU)

    model.eval()
    test_loss = 0.0

    y_pred = []
    y = []

    gpu = next(model.parameters()).device

    with torch.no_grad():
        for index, batch in enumerate(test_loader, 1):
            inputs, labels, lengths = batch
            inputs = inputs[:,:,:128]
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            valence, arousal, danceability = model(inputs)
            y_preds = [valence, arousal, danceability]
            y_preds = torch.cat(y_preds, dim=1)
            
            loss = criterion(y_preds, labels)

            y_pred.append(y_preds.cpu().numpy())
            y.append(labels.cpu().numpy())
            
            test_loss += loss.data.item()

    return y, y_pred

y, y_pred = evaluate_multitask(cnn_multi, test_loader_multi)
gt = np.array([])
for b in y:
    gt = np.append(gt,b)

pr = np.array([])
for b in y_pred:
    pr = np.append(pr,b)

from scipy.stats import spearmanr
print("Multitask CNN Spearman correlation: %f" %spearmanr(gt, pr)[0])