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
	train_data = lstm.FrameLevelDataset(X_train,y_train)
	val_data = lstm.FrameLevelDataset(X_dev,y_dev)

	#Define dataloader
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
	validation_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True, num_workers=4)

	#Define our LSTM
	nn = lstm.BasicLSTM(input_dim = 13, rnn_size = 10, output_dim = 10, num_layers = 2,bidirectional=True, dropout=0.2)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(nn.parameters(), lr=0.005,weight_decay = 1e-5)#we use weight_dacay for l2 regularization

	#Training
	epochs = 60
	train_losses = []
	val_losses = []
	val_curr = 0
	val_prev = 0
	counter = 0
	for i in range(epochs):
		nn.train()
		for index, batch in enumerate(train_loader, 1):#train nn with every batch of trainig set
			inputs, labels, lengths = batch

			optimizer.zero_grad()
			
			out = nn(inputs, lengths)
			train_loss = criterion(out, labels)
			train_loss.backward()

			
			optimizer.step()
			train_losses.append(train_loss.item())
			#print('Epoch {}, Training loss:{}'.format(i, train_loss.item()))
		for index, batch in enumerate(train_loader, 1):#we eavaluate the model with each batch of validation set
			val_inputs, val_labels, val_lengths = batch
			val_out = nn(val_inputs, val_lengths)
			val_loss = criterion(val_out, val_labels)
			val_losses.append(val_loss.item())
			print('Epoch {}, Validation loss:{}'.format(i, val_loss.item()))
		val_curr = val_loss.item()
		if(np.abs(val_curr-val_prev)<0.12): #implementation of early stopping
			counter = counter + 1
		else: 
			counter = 0
		epoxi = i+1
		if (counter == 10):
			break

	torch.save(nn.state_dict(),'val_training_50ep_dropout_early_bd.pt')


	x = np.linspace(0,epoxi,epoxi*16)
	plt.plot(x,train_losses,label='training loss')
	plt.plot(x,val_losses,label='Validation_loss')
	plt.xlabel('epochs')
	plt.ylabel('training and validation loss_dropout')
	plt.savefig('../plots/training_val_loss_60ep_dropout_early_bd')
	#plt.show()
