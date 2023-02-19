import os
from glob import glob
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pomegranate import *
import itertools
from lib2 import *
import pickle

#Step 9, 10, 11, 12
X_train, X_test, y_train, y_test, spk_train, spk_test = parser(sys.argv[1], n_mfcc=13)
# X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, stratify=y_train, random_state=42, test_size=0.2)
scale_fn = make_scale_fn(X_train + X_dev + X_test)
scale_fn = make_scale_fn(X_train + X_dev)
scale_fn = make_scale_fn(X_train)
X_train = scale_fn(X_train)
X_dev = scale_fn(X_dev)
X_test = scale_fn(X_test)

				# X_train = n_seq x seq_len x n_mfcc
				# we are converting into:
				# [(n_seq * seq_len) x n_mfcc) ]
#indices of each digit in X_train
digit_indices = [ np.where(np.array(y_train)==i)[0] for i in range(10)]
#MFCCs of each digit in X_train
digit_feats = []
for d in range(10):
	curr_digit_feats = np.zeros((1,13))
	for ind in digit_indices[d]:
		curr_digit_feats = np.concatenate((curr_digit_feats, X_train[ind]), axis=0)
	curr_digit_feats = curr_digit_feats[1:]
	digit_feats.append(curr_digit_feats)
#finally digit_feats is a list with len 10, with each element a (XXXX, 13) ndarray representing MFCCs of each digit
	
states = [1,2] # the number of HMM states
mixtures = [2,3,5] # the number of Gaussians
pairs = []
for state in states:
	for mixture in mixtures:
		pairs.append([state, mixture])

gmm = True # whether to use GMM or plain Gaussian
tryy = 0
accs = []
print('Parameter tuning on dev set:')
for n_states, n_mixtures in pairs:
	tryy += 1
	print('Try no.', tryy)

	models = [] # list with our GMM-HMM models for each digit
	for dig in range(10):
		dists = [] # list of probability distributions for the HMM states
		for i in range(n_states):
			if gmm:
				a = GeneralMixtureModel.from_samples(distributions=MultivariateGaussianDistribution, n_components=n_mixtures, X=digit_feats[dig])
			else:
				a = MultivariateGaussianDistribution.from_samples(digit_feats[dig])
			dists.append(a)

		#initialize transition matrix
		trans_mat = np.zeros([n_states,n_states]) # your transition matrix
		for i in range(0,n_states):
			if i==n_states-1:
				trans_mat[i][i]=1
			else:
				trans_mat[i][i]=0.5
				trans_mat[i][i+1]=0.5

		#initialize start prob matrix and end prob matrix
		starts = [1]+[0]*(n_states-1)
		ends = [0]*(n_states-1)+[1]

		# Define the GMM-HMM
		model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, state_names=['s{}'.format(i) for i in range(n_states)])

		# Fit the model
		model.fit(digit_feats[dig], max_iterations=5)

		models.append(model)
		print('Model created for digit:', dig)
		
	#Predict samples from dev set
	corrects = 0
	for i in range(len(X_dev)):
		logps = []
		for model in models:
			logp, _ = model.viterbi(X_dev[i]) # Run viterbi algorithm and return log-probability
			logps.append(logp)
		logps = np.array(logps)
		prediction = np.argmax(logps)
		if prediction == y_dev[i]:
			corrects += 1
	acc = np.round(corrects/len(X_dev),4)*100
	accs.append(acc)
	print('Dev accuracy with {} HMM states and {} gaussian mix: {}%'.format(n_states, n_mixtures, acc))
accs = np.array(accs)
best_acc = np.max(accs)
best_try = np.argmax(accs)

print('Best accuracy ({}%) was achieved with {} HMM states and {} gaussian mix.'.format(best_acc, pairs[best_try][0], pairs[best_try][1]))
print('Evaluating best model on test set:')

n_states = pairs[best_try][0]
n_mixtures = pairs[best_try][1]
best_models = []
for dig in range(10):
	dists = []
	for i in range(n_states):
		if gmm:
			a = GeneralMixtureModel.from_samples(distributions=MultivariateGaussianDistribution, n_components=n_mixtures, X=digit_feats[dig])
		else:
			a = MultivariateGaussianDistribution.from_samples(digit_feats[dig])
		dists.append(a)

	trans_mat = np.zeros([n_states,n_states])
	for i in range(0,n_states):
		if i==n_states-1:
			trans_mat[i][i]=1
		else:
			trans_mat[i][i]=0.5
			trans_mat[i][i+1]=0.5

	starts = [1]+[0]*(n_states-1)
	ends = [0]*(n_states-1)+[1]

	model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, state_names=['s{}'.format(i) for i in range(n_states)])
	model.fit(digit_feats[dig], max_iterations=5)

	best_models.append(model)
	print('Best model created for digit:', dig)
		
#Predict samples from test set
corrects = 0
for i in range(len(X_test)):
	logps = []
	for model in best_models:
		logp, _ = model.viterbi(X_test[i]) # Run viterbi algorithm and return log-probability
		logps.append(logp)
	logps = np.array(logps)
	prediction = np.argmax(logps)
	if prediction == y_test[i]:
		corrects += 1
acc = np.round(corrects/len(X_test),4)*100
print('Test accuracy with {} HMM states and {} gaussian mix: {}%'.format(n_states, n_mixtures, acc))


#Step 13
#Confusion matrices
confusion_val=np.zeros([len(np.unique(y_dev)),len(np.unique(y_dev))])
corrects = 0
for i in range(len(X_dev)):
	logps = []
	for model in best_models:
		logp, _ = model.viterbi(X_dev[i]) # Run viterbi algorithm and return log-probability
		logps.append(logp)
	prediction = np.argmax(logps)
	confusion_val[y_dev[i]][prediction] +=1
	if prediction == y_dev[i]:
		corrects += 1
acc = np.round(corrects/len(X_dev),4)*100
plot_confusion_matrix(confusion_val.astype(int),[0,1,2,3,4,5,6,7,8,9], title='Val Confusion Matrix')
plt.savefig('../plots/conf_mat_val.png')

confusion_test=np.zeros([len(np.unique(y_test)),len(np.unique(y_test))])
corrects = 0
for i in range(len(X_test)):
	logps = []
	for model in best_models:
		logp, _ = model.viterbi(X_test[i]) # Run viterbi algorithm and return log-probability
		logps.append(logp)
	prediction = np.argmax(logps)
	confusion_test[y_test[i]][prediction] +=1
	if prediction == y_test[i]:
		corrects += 1
acc = np.round(corrects/len(X_test),4)*100
plot_confusion_matrix(confusion_test.astype(int),[0,1,2,3,4,5,6,7,8,9], title='Test Confusion Matrix')
plt.savefig('../plots/conf_mat_test.png')

