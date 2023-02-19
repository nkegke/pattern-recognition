#Imports
import numpy as np
import lib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, ShuffleSplit, cross_val_score, KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import VotingClassifier, BaggingClassifier
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

#Pre-Lab

#Step 1
train_data = np.loadtxt('train.txt')
y_train = train_data[:,0] #labels
X_train = train_data[:,1:] #samples
test_data = np.loadtxt('test.txt')
y_test = test_data[:,0]
X_test = test_data[:,1:]
# print(X_train.shape)
# print(X_test.shape)


#Step 2
lib.show_sample(X_train, 131)
plt.savefig('2.png')


#Step 3
plt.figure()
lib.plot_digits_samples(X_train, y_train)
plt.savefig('3.png')


#Step 4
mean = lib.digit_mean_at_pixel(X_train, y_train, 0, (10, 10))
print('Mean value of pixel (10,10) in digit 0:', mean)


#Step 5
var = lib.digit_variance_at_pixel(X_train, y_train, 0, pixel=(10, 10))
print('Variance of pixel (10,10) in digit 0:', var)


#Step 6
mean = lib.digit_mean(X_train, y_train, 0)
var = lib.digit_variance(X_train, y_train, 0)


#Step 7
plt.figure()
plt.title('Digit 0 drawn by mean')
plt.imshow(mean, cmap='gray')
plt.savefig('7.png')


#Step 8
plt.figure()
plt.title('Digit 0 drawn by variance')
plt.imshow(var, cmap='gray')
plt.savefig('8.png')


#Step 9 a)
#we repeat steps 6,7,8 for all digits 
digit_means = [] #mean of each digit
digit_vars = [] #var of each digit
for i in range(10):
    digit_means.append(lib.digit_mean(X_train, y_train, i))
    digit_vars.append(lib.digit_variance(X_train, y_train, i))


#		b)
#plot the mean values and the variances of all digits
fig, axs = plt.subplots(2,5)	#Plotting the sample of each digit
fig.suptitle(' Means of digits ', fontsize=30)
for i in range(2):
    for j in range(5):
        img = digit_means[i*5+j].reshape((16,16))
        axs[i][j].imshow(img, cmap='gray')
        axs[i][j].set_title('digit ' + str(i*5+j))
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.savefig('9b1.png')


plt.figure()
fig, axs = plt.subplots(2,5)	#Plotting the sample of each digit
fig.suptitle(' Variations of digits ', fontsize=30)
for i in range(2):
    for j in range(5):
        img = digit_vars[i*5+j].reshape((16,16))
        axs[i][j].imshow(img, cmap='gray')
        axs[i][j].set_title('digit ' + str(i*5+j))
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.savefig('9b2.png')


#Step 10
distances = []
for i in range(10):
    distance = lib.euclidean_distance(X_test[101],digit_means[i])
    distances.append(distance)

min_dist = np.argmin(np.array(distances))
plt.figure()
plt.imshow(X_test[101].reshape((16,16)), cmap='gray')
plt.title('Real label: ' + str(int(y_test[101])) + ', Classified as: ' + str(min_dist))
plt.savefig('10.png')


#Step 11 a)
#using custom euclidean classifier
preds = lib.euclidean_distance_classifier(X_test,digit_means)


#		 b)
accuracy = [1 if preds[i] == int(y_test[i]) else 0 for i in range(preds.shape[0])]
acc = np.sum(accuracy)/len(accuracy)
print('Custom Euclidean Classifier Accuracy: ', acc)


#Step 12
#Creating sklearn-like Euclidean Classifier
euc_clf = lib.EuclideanDistanceClassifier()
euc_clf.fit(X_train, y_train)
euc_score = euc_clf.score(X_test, y_test)


#Step 13 a) 5-fold cross validation
euc_clf = lib.EuclideanDistanceClassifier()
# Concatenating train and test data to create the dataset to fold
X_new = np.concatenate((X_train, X_test), axis=0)
y_new = np.concatenate((y_train, y_test), axis=0)
lib.evaluate_classifier(euc_clf, X_new, y_new, folds=5)


#		 b) Plot decision regions
# Reduce dimensions from 256 to 2 with PCA
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_2D = pca.transform(X_train)
X_test_2D = pca.transform(X_test)

euc_clf = lib.EuclideanDistanceClassifier()
euc_clf.fit(X_train_2D, y_train)
labels = [i for i in range(10)]
plt.figure()
#this function is implemented on the bottom of lib.py
lib.plot_clf(euc_clf, X_test_2D, y_test, labels)
plt.savefig('13b.png')

#		 c) Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(
    lib.EuclideanDistanceClassifier(), X_new, y_new, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(.1, 1.0, 5))
plt.figure()
#this function is implemented on the bottom of lib.py
lib.plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.6, 1))
plt.savefig('13c.png')



############### END OF PRE-LAB #####################



#Step 14
#A-priori probabilities
prior = lib.calculate_priors(y_train, 0)
plt.figure()
plt.bar([i for i in range(10)], prior)
plt.xticks([i for i in range(10)])
plt.title('Class priors')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.savefig('14.png')

#Step 15
nvb = lib.CustomNBClassifier()
nvb.fit(X_train,y_train)
nvb.score(X_test,y_test)


clf = GaussianNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


score_clf = cross_val_score(clf, X_train, y_train, cv=KFold(n_splits=5), scoring='accuracy')
print("Scikit NB CV score = %f +-%f" % (np.mean(score_clf), np.std(score_clf)))

score_nvb = cross_val_score(nvb, X_train, y_train, cv=KFold(n_splits=5), scoring='accuracy')
print("Custom NB CV score = %f +-%f" % (np.mean(score_nvb), np.std(score_nvb)))


#Step 16
#compare with variance set to 1 for all pixels
nvb2 = lib.CustomNBClassifier()
nvb2.fit(X_train,y_train,flag = True)
nvb2.score(X_test,y_test)


#Step 17
#test KNN and SVM alongside naive bayes
neigh = KNeighborsClassifier()
neigh.fit(X_train,y_train)
neigh.score(X_test,y_test)

#SVM
clf_svm = svm.SVC(decision_function_shape='ovo')
clf_svm.fit(X_train,y_train)
clf_svm.score(X_test,y_test)

score_neigh = cross_val_score(neigh, X_train, y_train, cv=KFold(n_splits=5), scoring='accuracy')
print("KNN CV score = %f +-%f" % (np.mean(score_neigh), np.std(score_neigh)))
score_clf_svm = cross_val_score(clf_svm, X_train, y_train, cv=KFold(n_splits=5), scoring='accuracy')
print("SVM CV score = %f +-%f" % (np.mean(score_clf_svm), np.std(score_clf_svm)))

#different kernel svm
svm_scnd_lnr = svm.SVC(kernel='linear',decision_function_shape='ovo')
svm_scnd_lnr.fit(X_train,y_train)
score_scnd_lnr = cross_val_score(svm_scnd_lnr, X_train, y_train, cv=KFold(n_splits=5), scoring='accuracy')
print("CV score_svm second linear = %f +-%f" % (np.mean(score_scnd_lnr), np.std(score_scnd_lnr)))
svm_scnd_lnr.score(X_test,y_test)

#Step 18
#first plot confusion matrix of previous classifiers
class_names = [i for i in range(10)]

# Plots for Confusion Matrices
############ WARNING: Only runs on cloud services ###############
############ and not locally for some reason      ###############
# disp = ConfusionMatrixDisplay.from_estimator(clf_svm, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
# disp.ax_.set_title('SVM confusion matrix')
# plt.savefig('18a.png')
# 
# disp = ConfusionMatrixDisplay.from_estimator(neigh,X_test,y_test,display_labels=class_names,cmap=plt.cm.Blues)
# disp.ax_.set_title('KNN confusion matrix')
# plt.savefig('18b.png')
# 
# disp = ConfusionMatrixDisplay.from_estimator(clf,X_test,y_test,display_labels=class_names,cmap=plt.cm.Blues)
# disp.ax_.set_title('NB confusion matrix')
# plt.savefig('18c.png')

#Step 18 a)
eclf1 = VotingClassifier(estimators=[('lr', clf_svm), ('rf', neigh), ('gnb', clf)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
eclf1.score(X_test,y_test)

#		 b)
bagging = BaggingClassifier(base_estimator=clf_svm, n_estimators=10, random_state=0).fit(X_train, y_train)
bagging.score(X_test,y_test)                        
                        
############### BONUS PART #####################

#Step 19

#a)
#class to load our data to PyTorch through Dataloader
#inherits from Dataset class
class Digits(Dataset):
	
	def __init__(self, X, y, trans=None):
		self.data = list(zip(X, y))
		self.trans = trans
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if self.trans is not None:
			return self.trans(self.data[idx])
		else:
			return self.data[idx]


#b)
class LinearWActivation(nn.Module): 
	def __init__(self, in_features, out_features, activation):
		super(LinearWActivation, self).__init__()
		
		#fully connected   
		self.f = nn.Linear(in_features, out_features)
		
		#activation function
		if activation == 'relu':
			self.a = nn.ReLU()
		elif activation == 'tanh':
			self.a = nn.Tanh()

	#forward pass
	def forward(self, x): 
		return self.a(self.f(x))


class NeuralNetwork(nn.Module):
	def __init__(self, layers, n_features, n_classes, activation):
		'''
		Args:
			layers (list): a list of the number of consecutive layers
			n_features (int):  the number of input features
			n_classes (int): the number of output classes
			activation (str): type of non-linearity to be used
		'''
		super(NeuralNetwork, self).__init__()
		
		layers_in = [n_features] + layers # list concatenation
		layers_out = layers + [n_classes]
		
      	#list comprehension, calling the above class to create our layers
		self.f = nn.Sequential(*[
      		LinearWActivation(in_feats, out_feats, activation=activation)
      		for in_feats, out_feats in zip(layers_in, layers_out)
      	])
		# final classification layer is always a linear mapping
		self.clf = nn.Linear(n_classes, n_classes)
	
	def forward(self, x):
		y = self.f(x)
		return self.clf(y)


#Training
EPOCHS = 20
BATCH_SZ = 32
n_classes = len(set(y_train)) #convert list to set so duplicates disappear
criterion = nn.CrossEntropyLoss() #loss function
learning_rate = 1e-2 #for gradient descent


y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_data = Digits(X_train, y_train) #list of tuples:(array, label)
test_data = Digits(X_test, y_test)

train_dl = DataLoader(train_data, batch_size=BATCH_SZ, shuffle=True)
test_dl = DataLoader(test_data, batch_size=BATCH_SZ, shuffle=True)

#Hyperparameters to experiment with
hidden_layers = [[50], [100], [100, 50], [128, 64]]
activations = ['relu', 'tanh']


for layers in hidden_layers:
	for activation in activations:
		
		#define network architecture
		net = NeuralNetwork(layers, X_train.shape[1], n_classes, activation)		
		#print(f"The network architecture is: \n {net}")
		
		#feed the optimizer with the network parameters
		optimizer = optim.SGD(net.parameters(), lr=learning_rate)
		
		net.train() #gradients "on"
		for epoch in range(EPOCHS):
			running_average_loss = 0
			for i, data in enumerate(train_dl): # loop through batches
				X_batch, y_batch = data # get the features and labels
				X_batch = X_batch.to(torch.float)
				optimizer.zero_grad()
				out = net(X_batch) # forward pass
				loss = criterion(out, y_batch) # compute per batch loss 
				loss.backward() # compute gradients based on the loss function
				optimizer.step() # update weights 
				running_average_loss += loss.detach().item()

		net.eval() # turn off batchnorm/dropout ...
		acc = 0
		n_samples = 0
		with torch.no_grad():
			for i, data in enumerate(test_dl):
				X_batch, y_batch = data # test data and labels
				X_batch = X_batch.to(torch.float)
				out = net(X_batch) # get net's predictions
				val, y_pred = out.max(1) # argmax since output is a prob distribution
				acc += (y_batch == y_pred).sum().detach().item() # get accuracy
				n_samples += X_batch.size(0)

		accuracy = round(acc / n_samples, 3)
		print('Accuracy with {} hidden layers, {} neurons, {} activation : {}'.format(len(layers), layers, activation, accuracy))


#c)
class scikitNN(BaseEstimator, ClassifierMixin):
	
	def __init__(self, layers, n_features, n_classes, activation, batch_size):
		self.input_dim = n_features
		self.output_dim = n_classes
		self.layers = layers
		self.activation = activation
		self.batch_size = batch_size
		self.model = NeuralNetwork(self.layers, self.input_dim, self.output_dim, self.activation)
		
	def fit(self, X, y):
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
		
		train_data = Digits(X_train, y_train) #list of tuples:(array, label)
		train_dl = DataLoader(train_data, self.batch_size, shuffle=True)
		
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
		
		self.model.train()
		for epoch in range(20):
			running_average_loss = 0
			for i, data in enumerate(train_dl):
				X_batch, y_batch = data
				X_batch = X_batch.to(torch.float)
				optimizer.zero_grad()
				out = self.model(X_batch)
				loss = criterion(out, y_batch)
				loss.backward()
				optimizer.step()
				running_average_loss += loss.detach().item()
		
		#Evaluate the model on the validation set
		val_data = Digits(X_val, y_val)
		val_dl = DataLoader(val_data, self.batch_size, shuffle=True)
		self.model.eval()
		acc = 0
		n_samples = 0
		with torch.no_grad():
			for i, data in enumerate(val_dl):
				X_batch, y_batch = data
				X_batch = X_batch.to(torch.float)
				out = self.model(X_batch) # get net's predictions
				val, y_pred = out.max(1) # argmax since output is a prob distribution
				acc += (y_batch == y_pred).sum().detach().item() # get accuracy
				n_samples += X_batch.size(0)

		val_accuracy = round(acc / n_samples, 3)
		return val_accuracy
		
	def score(self, X, y):
		test_data = Digits(X, y)
		test_dl = DataLoader(test_data, self.batch_size, shuffle=True)

		self.model.eval() # turn off batchnorm/dropout ...
		acc = 0
		n_samples = 0
		with torch.no_grad():
			for i, data in enumerate(test_dl):
				X_batch, y_batch = data # test data and labels
				X_batch = X_batch.to(torch.float)
				out = self.model(X_batch) # get net's predictions
				val, y_pred = out.max(1) # argmax since output is a prob distribution
				acc += (y_batch == y_pred).sum().detach().item() # get accuracy
				n_samples += X_batch.size(0)

		test_accuracy = round(acc / n_samples, 3)
		return test_accuracy



my_net = scikitNN([256, 128, 64], 256, 10, 'relu', 32)
val_accuracy = my_net.fit(X_train, y_train)
print('Validation set accuracy with {} hidden layers, {} neurons, {} activation : {}'.format(len(my_net.layers), my_net.layers, my_net.activation, val_accuracy))

#d)
score = my_net.score(X_test, y_test)
print('Test set accuracy with {} hidden layers, {} neurons, {} activation : {}'.format(len(my_net.layers), my_net.layers, my_net.activation, score))