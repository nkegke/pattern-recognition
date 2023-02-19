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

def parse_free_digits(directory):
	# Parse relevant dataset info
	files = glob(os.path.join(directory, "*.wav"))
	fnames = [f.split("/")[1].split(".")[0].split("_") for f in files]
	ids = [f[2] for f in fnames]
	y = [int(f[0]) for f in fnames]
	speakers = [f[1] for f in fnames]
	_, Fs = librosa.core.load(files[0], sr=None)
	
	def read_wav(f):
		wav, _ = librosa.core.load(f, sr=None)
		return wav

	# Read all wavs
	wavs = [read_wav(f) for f in files]

	# Print dataset info
	print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

	return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=6, Fs=8000):
	# Extract MFCCs for all wavs
	window = 30 * Fs // 1000
	step = window // 2
	frames = [
		librosa.feature.mfcc(
			wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
		).T

		for wav in tqdm(wavs, desc="Extracting mfcc features...")
	]

	print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

	return frames


def split_free_digits(frames, ids, speakers, labels):
	print("Splitting in train test split using the default dataset split")
	# Split to train-test
	X_train, y_train, spk_train = [], [], []
	X_test, y_test, spk_test = [], [], []
	test_indices = ["0", "1", "2", "3", "4"]
	
	for idx, frame, label, spk in zip(ids, frames, labels, speakers):
		if str(idx) in test_indices:
			X_test.append(frame.astype(np.float64))
			y_test.append(label)
			spk_test.append(spk)
		else:
			X_train.append(frame.astype(np.float64))
			y_train.append(label)
			spk_train.append(spk)

	return X_train, X_test, y_train, y_test, spk_train, spk_test


def make_scale_fn(X_train):
	# Standardize on train data
	scaler = StandardScaler()
	scaler.fit(np.concatenate(X_train))
	def scale(X):
		scaled = []
		
		for frames in X:
			scaled.append(scaler.transform(frames))
		return scaled
	return scale


def parser(directory, n_mfcc=13):
	wavs, Fs, ids, y, speakers = parse_free_digits(directory)
	frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
	X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
		frames, ids, speakers, y
	)

	return X_train, X_test, y_train, y_test, spk_train, spk_test

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
                 
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
