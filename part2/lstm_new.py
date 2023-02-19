import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


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
		self.rnn = torch.nn.LSTM(input_size = input_dim, hidden_size = rnn_size, num_layers = num_layers, batch_first = True, bidirectional=self.bidirectional,dropout = dropout) #,dtype=torch.double
		self.linear = nn.Linear(self.feature_size, output_dim)

	def forward(self, x, lengths): 
		""" 
			x : 3D numpy array of dimension N x L x D
				N: batch index
				L: sequence index
				D: feature index
			lengths: N x 1
		 """
		hidden_dim = self.rnn_size
		if(self.bidirectional):
			ht_num_layers = self.num_layers*2 # for bidirectional we need double size of layers
		else:
			ht_num_layers = self.num_layers
		h_t = torch.zeros(ht_num_layers, x.shape[0], hidden_dim)
		c_t = torch.zeros(ht_num_layers, x.shape[0], hidden_dim)

		out, h_n = self.rnn(x, (h_t,c_t)) #forward pass for LSTM
		last_outputs = self.linear(self.last_timestep(out, lengths, self.bidirectional))

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
		# Index of the last output for each sequence.
		idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
											   outputs.size(2)).unsqueeze(1)
		return outputs.gather(1, idx).squeeze()