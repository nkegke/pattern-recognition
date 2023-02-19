import numpy as np
import matplotlib.pyplot as plt
import soundfile
from librosa.feature import mfcc
from librosa.feature import melspectrogram
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')


#Step 2
#Data parsing
import lib

#wavs: list with [sample_rate, time_series] for each wav file
#talkers: int list
#digits: string list
wavs, talkers, digits = lib.data_parser('./digits/')


#Step 3
MFCCs=[]
hop_sec = 0.01 # window step: 10msec
win_len_sec = 0.025 # window size: 25msec
sample_rate = 16000 # Hz
hop = int(sample_rate * hop_sec) # hop samples
win_len = int(sample_rate * win_len_sec) # window length samples
for wav in wavs:
    MFCC = mfcc(wav, sr=sample_rate, n_mfcc=13, hop_length=hop, win_length=win_len)
    MFCCs.append(MFCC)
#list of ndarrays of shape: (13, XX)

#deltas and delta-deltas
from librosa.feature import delta
deltas = []
ddeltas = []
for MFCC in MFCCs:
    deltas.append(delta(MFCC, order=1))
    ddeltas.append(delta(MFCC, order=2))


#Step 4
#n1 = 5, n2 = 4
fives = MFCCs[15:30]
fours = MFCCs[30:45]

##plot
plt.figure()
fig, axs = plt.subplots(3,5,figsize=(17,9))
for i in range(3):
    for j in range(5):
        axs[i][j].hist(fives[i*5+j][0])
plt.suptitle('histograms of number five from every speaker coeficient 0')
plt.savefig('../plots/hist5coef0.png')
plt.figure()
fig, axs = plt.subplots(3,5,figsize=(20,9))
for i in range(3):
    for j in range(5):
        axs[i][j].hist(fours[i*5+j][0])
plt.suptitle('histograms of number four from every speaker coeficient 0')
plt.savefig('../plots/hist4coef0.png')

plt.figure()
fig, axs = plt.subplots(3,5,figsize=(17,9))
for i in range(3):
    for j in range(5):
        axs[i][j].hist(fives[i*5+j][1])
plt.suptitle('histograms of number five from every speaker coeficient 1')
plt.savefig('../plots/hist5coef1.png')
plt.figure()
fig, axs = plt.subplots(3,5,figsize=(20,9))
for i in range(3):
    for j in range(5):
        axs[i][j].hist(fours[i*5+j][1])
plt.suptitle('histograms of number four from every speaker coeficient 1')
plt.savefig('../plots/hist4coef1.png')

MFSC5_1 = melspectrogram(wavs[15], sr=sample_rate, n_mels=13, hop_length=hop, win_length=win_len)
MFSC5_2 = melspectrogram(wavs[16], sr=sample_rate, n_mels=13, hop_length=hop, win_length=win_len)
MFSC4_1 = melspectrogram(wavs[30], sr=sample_rate, n_mels=13, hop_length=hop, win_length=win_len)
MFSC4_2 = melspectrogram(wavs[31], sr=sample_rate, n_mels=13, hop_length=hop, win_length=win_len)
lib.plotCorr([MFSC5_1, MFSC5_2, MFSC4_1, MFSC4_2], 'MFSCs')
plt.savefig('../plots/MFSCcorr.png')

#Plotting correlation of MFCCs to compare
lib.plotCorr([MFCCs[15], MFCCs[16], MFCCs[30], MFCCs[31]], 'MFFCs')
plt.savefig('../plots/MFFCcorr.png')


#Step 5
unified = []
for i in range(len(MFCCs)):
    mean_MFCC = np.mean(MFCCs[i], axis=1).reshape(1,13)
    mean_delta = np.mean(deltas[i], axis=1).reshape(1,13)
    mean_ddelta = np.mean(ddeltas[i], axis=1).reshape(1,13)
    std_MFCC = np.std(MFCCs[i], axis=1).reshape(1,13)
    std_delta = np.std(deltas[i], axis=1).reshape(1,13)
    std_ddelta = np.std(ddeltas[i], axis=1).reshape(1,13)
    unified.append(np.concatenate((mean_MFCC, mean_delta, mean_ddelta, std_MFCC, std_delta, std_ddelta), axis=1))
unified = np.array(unified).reshape(len(unified),unified[0].shape[1])

# unified : mean_MFCC | mean_delta | mean_ddelta | std_MFCC | std_delta | std_ddelta
# with shape: (133,78)

labels = [i+1 for i in range(9)]
plt.rcParams['figure.figsize'] = [15, 10]
lib.plot_2d(unified, digits, labels)
plt.title('Digits on 2D-space from first 2 dimensions')
plt.savefig('../plots/no_pca.png')


#Step 6
#Use PCA for a better projection on the 2D plane
pca2d = PCA(n_components=2)
projected_2d = pca2d.fit_transform(unified)
plt.rcParams['figure.figsize'] = [15, 10]
lib.plot_2d(projected_2d, digits, labels)
plt.title('Digits on 2D-space after PCA')
plt.savefig('../plots/pca_2d.png')

#Project on 3D space
pca3d = PCA(n_components=3)
projected_3d = pca3d.fit_transform(unified)
plt.rcParams['figure.figsize'] = [20, 20]
lib.plot_3d(projected_3d, digits, labels)
plt.title('Digits on 3D-space after PCA')
plt.savefig('../plots/pca_3d.png')

print('Percentage of initial deviation preserved with 2 PCs: ', pca2d.explained_variance_ratio_)
print('Percentage of initial deviation preversed with 3 PCs: ', pca3d.explained_variance_ratio_)

#Step 7
X_train, X_test, y_train, y_test = train_test_split(unified, digits, test_size=0.30, random_state=42)
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

nvb = lib.CustomNBClassifier()
nvb.fit(X_train_norm,y_train)
cnvb = nvb.score(X_test_norm,y_test)
print('Custom Gaussian Naive Bayes score on normalized data: ', cnvb)

scikit_nvb = GaussianNB()
scikit_nvb.fit(X_train, y_train)
snvb = scikit_nvb.score(X_test, y_test)
print('Scikit Gaussian Naive Bayes score: ', snvb)
scikit_nvb = GaussianNB()
scikit_nvb.fit(X_train_norm, y_train)
snvb = scikit_nvb.score(X_test_norm, y_test)
print('Scikit Gaussian Naive Bayes score on normalized data: ', snvb)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
sknn = knn.score(X_test, y_test)
print('Scikit K Nearest Neighbors (K=1) score: ', sknn)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_norm, y_train)
sknn = knn.score(X_test_norm, y_test)
print('Scikit K Nearest Neighbors (K=1) score on normalized data: ', sknn)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
sknn = knn.score(X_test, y_test)
print('Scikit K Nearest Neighbors (K=2) score: ', sknn)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_norm, y_train)
sknn = knn.score(X_test_norm, y_test)
print('Scikit K Nearest Neighbors (K=2) score on normalized data: ', sknn)

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
ssvm = svm.score(X_test, y_test)
print('Scikit SVM linear score: ', ssvm)

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_norm, y_train)
ssvm = svm.score(X_test_norm, y_test)
print('Scikit SVM linear score on normalized data: ', ssvm)

svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
ssvm = svm.score(X_test, y_test)
print('Scikit SVM rbf score: ', ssvm)

svm = SVC(kernel='poly', probability=True)
svm.fit(X_train, y_train)
ssvm = svm.score(X_test, y_test)
print('Scikit SVM poly score: ', ssvm)

mlp = MLPClassifier(hidden_layer_sizes=(50,25), activation='relu', max_iter=1000)
mlp.fit(X_train, y_train)
smlp = mlp.score(X_test, y_test)
print('Scikit MLP score: ', smlp)

mlp = MLPClassifier(hidden_layer_sizes=(50,25), activation='relu', max_iter=1000)
mlp.fit(X_train_norm, y_train)
smlp = mlp.score(X_test_norm, y_test)
print('Scikit MLP score on normalized data: ', smlp)


#Step 8
#PyTorch
f = 40
#distance between points: pi/5
t = np.linspace(0,2*np.pi,10)
sinseq = []
cosseq = []
# fig, axs = plt.subplots(10,2)
# fig.set_figheight(45)
# fig.set_figwidth(15)
for i in range(1000):
    A = np.random.rand()
    p = np.random.rand()*2*np.pi
    sinseq.append(A*np.sin(2*np.pi*f*t + p))
#     axs[i][0].plot(t, sinseq[i])
#     axs[i][0].set_title('Sine with amplitude: '+str(np.round(A,2))+'  and  phase: ' + str(np.round(p,2))+ ' rad')
    cosseq.append(A*np.cos(2*np.pi*f*t + p))
#     axs[i][1].plot(t, cosseq[i])
#     axs[i][1].set_title('Cosine with amplitude: '+str(np.round(A,2))+'  and  phase: ' + str(np.round(p,2)) + ' rad')
fig, axs = plt.subplots(2)
axs[0].plot(t, sinseq[999])
axs[0].set_title('Sine with amplitude: '+str(np.round(A,2))+'  and  phase: ' + str(np.round(p,2))+ ' rad')
axs[1].plot(t, cosseq[999])
axs[1].set_title('Cosine with amplitude: '+str(np.round(A,2))+'  and  phase: ' + str(np.round(p,2)) + ' rad')
plt.savefig('../plots/waveforms.png')

X = np.array(sinseq)
y = np.array(cosseq)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)


batch_size = X_train.shape[0] # how many mini-batches, each mini is a sine
seq_length = X_train.shape[1] # length of each mini-batch
input_sz = 1 # feed 1 point at a time
hidden_dim = 1

#Initialize our model
# rnn = torch.nn.RNN(input_size = input_sz, hidden_size = hidden_dim, num_layers = 1, batch_first = True).double()
rnn = torch.nn.LSTM(input_size = input_sz, hidden_size = hidden_dim, num_layers = 1, batch_first = True).double()
h_t = torch.zeros(1, X_train.shape[0], hidden_dim, dtype=torch.double)
c_t = torch.zeros(1, X_train.shape[0], hidden_dim, dtype=torch.double)

#input shape : (batch_size, seq_length, input_size)
X_train = X_train.view(batch_size, seq_length, input_sz)
X_test = X_test.view(X_test.shape[0], seq_length, input_sz)
y_train = y_train.view(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.view(y_test.shape[0], y_test.shape[1], 1)

#out shape : (batch_size, seq_length, hidden_dim)
#h_n shape : (num_layers, batch_size, hidden_dim)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
epochs = 4500

for i in range(epochs):
	optimizer.zero_grad()
	
# 	out, h_n = rnn(X_train) #forward pass for RNN
	out, h_n = rnn(X_train, (h_t,c_t)) #forward pass for LSTM

	#############	#############	#############	#############	#############
	#If hidden_dim > 1 :
	#This is what they say on a github issue, but performs worse!!
# 	output_layer = torch.nn.Linear(hidden_dim, 1).double() #convert hidden_dim size
# 	out = output_layer(out)								   #to target output size : 1
	#############	#############	#############	#############	#############

	loss = criterion(out, y_train) #calculate loss
	loss.backward() #back propagation
	if i%300==0:
		print('Training loss:', loss.item())
	optimizer.step()

	with torch.no_grad():
		pred, h_next = rnn(X_test)
		loss = criterion(pred, y_test)
		if i%300==0:
			print('Test loss:', loss.item())

with torch.no_grad():
    pred, h_next = rnn(X_test.view(X_test.shape[0], seq_length, input_sz))

    	#############	#############	#############	#############
# output_layer = torch.nn.Linear(hidden_dim, 1).double() #convert hidden_dim size
# pred = output_layer(pred)					   		   #to target output size : 1
		#############	#############	#############	#############

fig = plt.figure(figsize=(25,10))
for i in range(15):
    fig.add_subplot(3, 5, i+1)
    plt.axis('off')
    output = pred[i*5].view(10).detach().numpy()
    plt.plot(np.arange(10), output, 'g')
    plt.plot(np.arange(10), y_test[i*5], 'r')
plt.show()
