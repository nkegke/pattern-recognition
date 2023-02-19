import os
import sys
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import parser2 as pr
import lstm_new as lstm
import torch
if __name__ == '__main__':
	X_train, X_test, y_train, y_test, spk_train, spk_test = pr.parser(sys.argv[1])
	X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train)
	scale_fn = pr.make_scale_fn(X_train + X_dev + X_test)
	scale_fn = pr.make_scale_fn(X_train + X_dev)
	scale_fn = pr.make_scale_fn(X_train)
	X_train = scale_fn(X_train)
	X_dev = scale_fn(X_dev)
	X_test = scale_fn(X_test)

	#Create dataset
	test_data = lstm.FrameLevelDataset(X_test,y_test)

	#Define dataloader
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4)

	#model = torch.nn.lstm(input_size = input_dim, hidden_size = rnn_size, num_layers = num_layers, batch_first = True, bidirectional=False,dropout = dropout)
	model = lstm.BasicLSTM(input_dim = 13,rnn_size = 10,output_dim = 10,num_layers = 2,bidirectional=True,dropout=0.2)
	model.load_state_dict(torch.load('val_training_50ep_dropout_early_bd.pt'))#choose the net you want to test
	model.eval()


	correct = 0
	total = 0
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for index, batch in enumerate(test_loader, 1):
			inputs, labels, lengths = batch
	        # samples, labels = data
	        # calculate outputs by running images through the network
			outputs = model(inputs,lengths)
	        # the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the test set: %d %%' % (
	    100 * correct / total))