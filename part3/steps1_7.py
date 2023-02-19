import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
from librosa import display
import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
import re

#STEP 1
# a)
# b)
spec = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/80238.fused.full.npy')
mel, chroma = spec[:128], spec[128:]

spec1 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/66082.fused.full.npy')
spec2 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/19263.fused.full.npy')

mel1, chroma1 = spec1[:128], spec1[128:]
mel2, chroma2 = spec2[:128], spec2[128:]

display.specshow(mel1)
plt.show()

display.specshow(mel2)
plt.show()

#STEP 2
# b)
spec_beat1 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/66082.fused.full.npy')
spec_beat2 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/19263.fused.full.npy')

mel_beat1, chroma_beat1 = spec_beat1[:128], spec_beat1[128:]
mel_beat2, chroma_beat2 = spec_beat2[:128], spec_beat2[128:]

display.specshow(mel_beat1)
plt.show()

display.specshow(mel_beat2)
plt.show()

#STEP 3
display.specshow(chroma1)
plt.show()

display.specshow(chroma2)
plt.show()

display.specshow(chroma_beat1)
plt.show()

display.specshow(chroma_beat2)
plt.show()

#STEP 4
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


dataset = SpectrogramDataset("/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms", class_mapping=class_mapping, train=True)
print(dataset[10])
print(f"Input: {dataset[10][0].shape}")
print(f"Label: {dataset[10][1]}")
print(f"Original length: {dataset[10][2]}")

plt.hist(dataset.labels, bins=10)
plt.xlabel('Labels')
plt.xticks(np.unique(dataset.labels))
plt.ylabel('Samples')
plt.title('Spectogram Dataset')
plt.show()

dataset_without_mapping = SpectrogramDataset("/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms", train=True)

plt.hist(dataset_without_mapping.labels)
plt.xlabel('Labels')
plt.xticks(np.unique(dataset_without_mapping.labels))
plt.ylabel('Samples')
plt.show()


#STEP 5
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay = 1e-5)

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
            y_preds = model(inputs, lengths)

            loss = criterion(y_preds, labels)
            loss.backward()

            optimizer.step()

            train_loss = loss.data.item()

            if epoch%100 == 0:
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
                y_preds = model(inputs, lengths)

                loss = criterion(y_preds, labels)
                loss.backward()

                optimizer.step()

                train_loss = loss.data.item()

            if epoch%3 == 0:
                print("Epoch %d with training loss: %f" %(epoch, train_loss))
            
            if epoch%25 == 0:

                for index, batch in enumerate(val_loader, 1):
                    val_loss = 0.0

                    gpu = next(model.parameters()).device

                    inputs, labels, lengths = batch

                    inputs = inputs.to(gpu)
                    labels = labels.to(gpu)
                    
                    y_preds = model(inputs, lengths)

                    loss = criterion(y_preds, labels)

                    val_loss = loss.data.item()

                print("Epoch %d with validation loss: %f" %(epoch, val_loss))

#5b)
#Define dataloaders
train_loader, val_loader = torch_train_val_split(dataset, batch_train=32, batch_eval=32, val_size=.2, shuffle=True, seed=42)

#Define LSTM
nn_to_overfit = BasicLSTM(input_dim = dataset.feat_dim, rnn_size = 32, output_dim = 10, num_layers = 1, bidirectional=True).double()

train(nn_to_overfit, train_loader, val_loader, 3500, overfit_batch=True)

#Save overfitted model
torch.save(nn_to_overfit, './overfitted_model')

#5c)
#Define dataloaders
mel_specs = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_mel, val_loader_mel = torch_train_val_split(mel_specs, 32 ,32, val_size=.33)


# #Define LSTM
my_nn_spec = BasicLSTM(input_dim = dataset.feat_dim, rnn_size = 32, output_dim = 10, num_layers = 4, bidirectional=True).double()

train(my_nn_spec, train_loader, val_loader, 300)

trained_model =  BasicLSTM(input_dim = dataset.feat_dim, rnn_size = 32, output_dim = 10, num_layers = 4, bidirectional=True)#.double()

#Save model trained on spectograms
torch.save(my_nn_spec, './trained_model_on_spectograms')


#5d)
beat_mel_specs = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_beat_mel, val_loader_beat_mel = torch_train_val_split(beat_mel_specs, 32 ,32, val_size=.33)


nn_beat_spec = BasicLSTM(input_dim = beat_mel_specs.feat_dim, rnn_size = 32, output_dim = 10, num_layers = 4, bidirectional=True).double()
train(nn_beat_spec, train_loader_beat_mel, val_loader_beat_mel, 400)

#Save model trained on beat spectograms
torch.save(nn_beat_spec, './trained_model_on_beat_spectograms')


#5e)
beat_chroma = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_chromagram)
train_loader_beat_chroma, val_loader_beat_chroma = torch_train_val_split(beat_chroma, 32 ,32, val_size=.33)

nn_beat_chroma = BasicLSTM(input_dim = beat_chroma.feat_dim, rnn_size = 32, output_dim = 10, num_layers = 4, bidirectional=True).double()
train(nn_beat_chroma, train_loader_beat_chroma, val_loader_beat_chroma, 400)

# Save model trained on beat chromagrams
torch.save(nn_beat_chroma, './trained_model_on_beat_chromagrams')

#5z)
# specs_fused = SpectrogramDataset(
#          '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
#          train=True,
#          class_mapping=class_mapping,
#          max_length=-1,
#          read_spec_fn=read_fused_spectrogram)
# train_loader_fused, val_loader = torch_train_val_split(specs_fused, 32 ,32, val_size=.33)
# test_loader_fused = SpectrogramDataset(
#          '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
#          train=False,
#          class_mapping=class_mapping,
#          max_length=-1,
#          read_spec_fn=read_fused_spectrogram)

#STEP 6
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def evaluate(model, test_loader):
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0

    y_pred = []
    y = []

    gpu = next(model.parameters()).device

    with torch.no_grad():
        for index, batch in enumerate(test_loader, 1):
            inputs, labels, lengths = batch

            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            y_preds = model(inputs, lengths)  # EX9
            loss = criterion(y_preds, labels)

            #prediction
            y_preds_arg = torch.argmax(y_preds, dim=1)

            y_pred.append(y_preds_arg.cpu().numpy())
            y.append(labels.cpu().numpy())

            test_loss += loss.data.item()

    return y, y_pred

#beat mel
beat_test_set = SpectrogramDataset("/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat", class_mapping=class_mapping)
test_loader_beat_mel = DataLoader(beat_test_set, batch_size=32)
y, y_pred = evaluate(nn_beat_spec, test_loader_beat_mel)

gt = np.array([])
for b in y:
    gt = np.append(gt,b)

pr = np.array([])
for b in y_pred:
    pr = np.append(pr,b)

print("Accuracy:" , accuracy_score(gt, pr))
print(classification_report(gt, pr))

#mel
test_set = SpectrogramDataset("/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms", class_mapping=class_mapping)
test_loader_mel = DataLoader(test_set, batch_size=32)
y, y_pred = evaluate(my_nn_spec, test_loader_mel)

gt = np.array([])
for b in y:
    gt = np.append(gt,b)

pr = np.array([])
for b in y_pred:
    pr = np.append(pr,b)

print("Accuracy:" , accuracy_score(gt, pr))
print(classification_report(gt, pr))


#beat chroma
test_beat_chroma = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',train=False,class_mapping=class_mapping,max_length=-1,read_spec_fn=read_chromagram)
test_loader_beat_chroma = DataLoader(test_beat_chroma, batch_size=32)
y, y_pred = evaluate(nn_beat_chroma, test_loader_beat_chroma)

gt = np.array([])
for b in y:
    gt = np.append(gt,b)

pr = np.array([])
for b in y_pred:
    pr = np.append(pr,b)

print("Accuracy:" , accuracy_score(gt, pr))
print(classification_report(gt, pr))

#STEP 7
#a) on report

#b)
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
        self.fc1 = nn.Linear(13456, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
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
        return x


#c) explained in a) on report


def train(model, train_loader, val_loader, epochs, overfit_batch=False):
    #Use GPU
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4,weight_decay = 1e-5)

    if overfit_batch==True:
        batch = next(iter(train_loader)) #train with only one batch
        losses = []
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
            losses.append(train_loss)
            if epoch%5 == 0:
                print("Epoch %d with loss: %f" %(epoch, train_loss))
        val_losses = None
    else:
        losses = []
        val_losses = []
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

                loss = criterion(y_preds, labels)
                loss.backward()

                optimizer.step()

                train_loss = loss.data.item()
                losses.append(train_loss)
            if True:
                print("Epoch %d with training loss: %f" %(epoch, train_loss))
            
            if True:

                for index, batch in enumerate(val_loader, 1):
                    val_loss = 0.0

                    gpu = next(model.parameters()).device

                    inputs, labels, lengths = batch

                    inputs = inputs.to(gpu)
                    labels = labels.to(gpu)

                    #optimizer.zero_grad()
                    y_preds = model(inputs)

                    loss = criterion(y_preds, labels)

                    #optimizer.step()

                    val_loss = loss.data.item()
                    val_losses.append(val_loss)
                    #if epoch%100 == 0:
                print("Epoch %d with validation loss: %f" %(epoch, val_loss))
    return losses, val_losses


#d)
net = Net().double()
overfit_losses = train(net, train_loader_beat_mel, val_loader_beat_mel, 50, overfit_batch=True)
plt.plot(overfit_losses)
plt.title('Overfit CNN loss')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.show()

#e)
beat_mel_specs = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_beat_mel, val_loader_beat_mel = torch_train_val_split(beat_mel_specs, 32 ,32, val_size=.33)
test_beat_mel = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
test_loader_beat_mel = DataLoader(test_beat_mel, batch_size=32)

net = Net().double()
CNN_mel_losses, CNN_val_losses = train(net, train_loader_beat_mel, val_loader_beat_mel, 50, overfit_batch=False)
plt.plot(CNN_mel_losses)
plt.plot(CNN_val_losses)
plt.title('CNN loss')
plt.xlabel('Epochs')
plt.legend(['Training loss', 'Validation loss'])
plt.show()


from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
def evaluate(model, test_loader):
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(GPU)

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0

    y_pred = []
    y = []

    gpu = next(model.parameters()).device

    with torch.no_grad():
        for index, batch in enumerate(test_loader, 1):
            inputs, labels, lengths = batch

			#Zeropadding so test set has same dimension with training set
            padder = torch.zeros((inputs.shape[0],13,128)).double()
            inputs = torch.cat((inputs,padder), dim=1)

            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

            #prediction
            y_preds_arg = torch.argmax(y_preds, dim=1)

            y_pred.append(y_preds_arg.cpu().numpy())
            y.append(labels.cpu().numpy())

            test_loss += loss.data.item()

    return y, y_pred

test_beat_mel = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectogram)
test_loader_beat_mel = DataLoader(test_beat_mel, batch_size=32)

y, y_pred = evaluate(net, test_loader_beat_mel)

gt = np.array([])
for b in y:
    gt = np.append(gt,b)

pr = np.array([])
for b in y_pred:
    pr = np.append(pr,b)

print("Accuracy:" , accuracy_score(gt, pr))
print(classification_report(gt, pr))