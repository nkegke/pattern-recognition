import os
from librosa import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
import pandas as pd
import torch

def data_parser(directory):
    wavs = []
    talkers = []
    digits = []
    number_set = set([str(i) for i in range(1, 10)])
    digit_dict = {'one': int(1), 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                  'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
    for filename in sorted(os.listdir(directory)):
        wavs.append(load('./digits/' + filename, sr=16000)[0])
        for char in filename:
            if char in number_set:
                split = filename.split(char)
                break 
        digits.append(digit_dict[split[0]])
        talkers.append(int(char))
    return wavs, talkers, np.array(digits)



def plot_2d(X, y, labels):
    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]

    ones = ax.scatter(
        X0[y == 1], X1[y == 1],
        c='red', label=labels[0], 
        s=60, alpha=0.9, edgecolors='k')
    twos = ax.scatter(
        X0[y == 2], X1[y == 2],
        c='yellow', label=labels[1], 
        s=60, alpha=0.9, edgecolors='k')
    threes = ax.scatter(
        X0[y == 3], X1[y == 3],
        c='white', label=labels[2], 
        s=60, alpha=0.9, edgecolors='k')
    fours = ax.scatter(
        X0[y == 4], X1[y == 4],
        c='black', label=labels[3], 
        s=60, alpha=0.9, edgecolors='k')
    fives = ax.scatter(
        X0[y == 5], X1[y == 5],
        c='orange', label=labels[4], 
        s=60, alpha=0.9, edgecolors='k')
    sixes = ax.scatter(
        X0[y == 6], X1[y == 6],
        c='green', label=labels[5], 
        s=60, alpha=0.9, edgecolors='k')
    sevens = ax.scatter(
        X0[y == 7], X1[y == 7],
        c='pink', label=labels[6], 
        s=60, alpha=0.9, edgecolors='k')
    eights = ax.scatter(
        X0[y == 8], X1[y == 8],
        c='gray', label=labels[7], 
        s=60, alpha=0.9, edgecolors='k')
    nines = ax.scatter(
        X0[y == 9], X1[y == 9],
        c='purple', label=labels[8], 
        s=60, alpha=0.9, edgecolors='k')
    
    ax.set_ylabel('2nd dimension')
    ax.set_xlabel('1st dimension')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()


def plot_3d(X, y, labels):
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    X0, X1, X2 = X[:, 0], X[:, 1], X[:, 2]

    ones = ax.scatter(
        X0[y == 1], X1[y == 1], X2[y == 1],
        c='red', label=labels[0], 
        s=60, alpha=0.9, edgecolors='k')
    twos = ax.scatter(
        X0[y == 2], X1[y == 2], X2[y == 2],
        c='yellow', label=labels[1], 
        s=60, alpha=0.9, edgecolors='k')
    threes = ax.scatter(
        X0[y == 3], X1[y == 3], X2[y == 3],
        c='white', label=labels[2], 
        s=60, alpha=0.9, edgecolors='k')
    fours = ax.scatter(
        X0[y == 4], X1[y == 4], X2[y == 4],
        c='black', label=labels[3], 
        s=60, alpha=0.9, edgecolors='k')
    fives = ax.scatter(
        X0[y == 5], X1[y == 5], X2[y == 5],
        c='orange', label=labels[4], 
        s=60, alpha=0.9, edgecolors='k')
    sixes = ax.scatter(
        X0[y == 6], X1[y == 6], X2[y == 6],
        c='green', label=labels[5], 
        s=60, alpha=0.9, edgecolors='k')
    sevens = ax.scatter(
        X0[y == 7], X1[y == 7], X2[y == 7],
        c='pink', label=labels[6], 
        s=60, alpha=0.9, edgecolors='k')
    eights = ax.scatter(
        X0[y == 8], X1[y == 8], X2[y == 8],
        c='gray', label=labels[7], 
        s=60, alpha=0.9, edgecolors='k')
    nines = ax.scatter(
        X0[y == 9], X1[y == 9], X2[y == 9],
        c='purple', label=labels[8], 
        s=60, alpha=0.9, edgecolors='k')
    
    ax.set_zlabel('3rd dimension')
    ax.set_ylabel('2nd dimension')
    ax.set_xlabel('1st dimension')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    ax.legend()


def digit_mean_at_index(X, y, digit, index):
    indexes = np.where(y==digit)[0]
    sum = 0
    for ind in indexes:
        sum += X[ind][index]
    return sum/len(indexes)


def digit_variance_at_index(X, y, digit, index):
    mean = digit_mean_at_index(X, y, digit, index)
    indexes = np.where(y==digit)[0]
    sum = 0
    for ind in indexes:
        sum += (X[ind][index]-mean)**2
    return sum/len(indexes)


def digit_mean(X, y, digit):
	#Calling digit_mean_at_pixel function for every index
	return np.array([digit_mean_at_index(X, y, digit, index=i) for i in range(X.shape[1])])#before X.shape we had 78


def digit_variance(X, y, digit):
	#Calling variance_at_pixel function for every index
	return np.array([digit_variance_at_index(X, y, digit, index=i) for i in range(X.shape[1])])

def calculate_priors(y):
	prior = np.array([np.where(y == i+1)[0].shape[0] for i in range(y.shape[0])])/y.shape[0]
	return prior

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        self.digit_means = None
        self.digit_vars = None
        self.y = None
        
    def fit(self, X, y,flag=False):
        #if flag is true then all variances are equal to one
        self.y = y
        self.digit_means = np.ndarray(shape=(y.shape[0],X.shape[1]))#9,78
        if flag:
            self.digit_vars = np.ones(shape=(y.shape[0],X.shape[1]))
        else:
            self.digit_vars = np.ndarray(shape=(y.shape[0],X.shape[1]))
        for i in range(9):
            self.digit_means[i] = (digit_mean(X, y, i+1))
            if not(flag):
                self.digit_vars[i] = (digit_variance(X, y, i+1))+0.05#add 10^(-6) because we divide these numbers as 0
        return self
        
    def predict(self, X):
        #naive bayes
		#P(yi|x1,x2,...,xn) = P(x1|yi)P(x2|yi)...P(xn|yi)*P(yi)
		#where p(yi|xi)~N(mu_i,sigma_i)
		#first calculate all dependent propabilities
        prior = calculate_priors(self.y)

        prop = np.ones(shape=(X.shape[0],self.y.shape[0],X.shape[1]))
        for i in range(X.shape[0]):#number of sample
            for j in range(self.y.shape[0]):#number of class
                prop[i][j] = stats.norm.pdf(X[i],self.digit_means[j],self.digit_vars[j]) #probability P(x_min|yi)

		#second calculate the naive bayes type
        mid_prop = np.ndarray(shape=(X.shape[0],self.y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.y.shape[0]):
                mid_prop[i][j] = np.prod(prop[i][j], where = prop[i][j] > 0, keepdims = True)*prior[j]

		#finally keep the class with biggest propability
        total_pred = np.argmax(mid_prop,1)+1
        return total_pred
    
    
    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = [1 if predictions[i] == int(y[i]) else 0 for i in range(predictions.shape[0])]
        acc = np.sum(accuracy)/len(accuracy)
        return acc

def plotCorr(coeffs, title):
	df5_1 = pd.DataFrame(coeffs[0].T)
	df5_2 = pd.DataFrame(coeffs[1].T)
	df4_1 = pd.DataFrame(coeffs[2].T)
	df4_2 = pd.DataFrame(coeffs[3].T)

	corr5_1 = df5_1.corr()
	corr5_2 = df5_2.corr()
	corr4_1 = df4_1.corr()
	corr4_2 = df4_2.corr()

	plt.figure()
	fig, axs = plt.subplots(2,2)
	plt.suptitle('Correlation of ' + str(title))
	axs[0][0].imshow(corr5_1)
	axs[0][0].set_title('Digit 5 - Speaker 1')
	axs[0][0].set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
	axs[0][0].set_xticks([])
	axs[0][1].imshow(corr5_2)
	axs[0][1].set_title('Digit 5 - Speaker 2')
	axs[0][1].set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
	axs[0][1].set_xticks([])
	axs[1][0].imshow(corr4_1)
	axs[1][0].set_title('Digit 4 - Speaker 1')
	axs[1][0].set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
	axs[1][0].set_xticks([])
	axs[1][1].imshow(corr4_2)
	axs[1][1].set_title('Digit 4 - Speaker 2')
	axs[1][1].set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
	axs[1][1].set_xticks([])



