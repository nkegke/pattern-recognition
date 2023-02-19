from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def show_sample(X, index):

	'''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		index (int): index of digit to show
	'''

	digit = X[index].reshape((16,16))
	plt.title('Sample ' + str(int(index)))
	plt.imshow(digit, cmap='gray')


def plot_digits_samples(X, y):

	'''Takes a dataset and selects one example from each label and plots it in subplots
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
	'''

	indexes = np.zeros(10) 			#Array to store indices of digits
	i = 0								
	while not(indexes.all()):		#Finding the first index of each digit
		if indexes[int(y[i])] == 0:
			indexes[int(y[i])] = int(i)
		i += 1

	indexes = indexes.reshape((2,5))
	fig, axs = plt.subplots(2,5)	#Plotting the sample of each digit
	for i in range(2):
		for j in range(5):
			img = X[int(indexes[i][j])].reshape((16,16))
			axs[i][j].imshow(img, cmap='gray')
			axs[i][j].set_title('sample ' + str(int(indexes[i][j])))
	plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):

	'''Calculates the mean for all instances of a specific digit at a pixel location
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
		pixels (tuple of ints): The pixels we need to select.
	Returns:
		(float): The mean value of the digits for the specified pixels
	'''
	if X.shape[1] == 2: # for 2-Dimensional data
		indexes = np.where(y==digit)[0]
		sum = 0
		for ind in indexes:
			sum += X[ind][pixel[0]+pixel[1]]
	else: 				# for 256-Dimensional data
		indexes = np.where(y==digit)[0]
		sum = 0
		for ind in indexes:
			sum += X[ind][16*pixel[0]+pixel[1]]
	return sum/len(indexes)




def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):

	'''Calculates the variance for all instances of a specific digit at a pixel location
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
		pixels (tuple of ints): The pixels we need to select
	Returns:

		(float): The variance value of the digits for the specified pixels
	'''

	mean = digit_mean_at_pixel(X, y, digit, pixel)
	indexes = np.where(y==digit)[0]
	sum = 0
	for ind in indexes:
		sum += (X[ind][16*pixel[0]+pixel[1]]-mean)**2
	return sum/len(indexes)




def digit_mean(X, y, digit):

	'''Calculates the mean for all instances of a specific digit
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
	Returns:
		(np.ndarray): The mean value of the digits for every pixel
	'''

	#Calling digit_mean_at_pixel function for every pixel
	if X.shape[1] == 2:	# for 2-Dimensional data
		return np.array([digit_mean_at_pixel(X, y, digit, pixel=(i, 0)) for i in range(2)])
	else:				# for 256-Dimensional data
		return np.array([digit_mean_at_pixel(X, y, digit, pixel=(i, j)) for i in range(16) for j in range(16)]).reshape((16,16))





def digit_variance(X, y, digit):

	'''Calculates the variance for all instances of a specific digit
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
	Returns:
		(np.ndarray): The variance value of the digits for every pixel
	'''

	#Calling variance_at_pixel function for every pixel
	return np.array([digit_variance_at_pixel(X, y, digit, pixel=(i, j)) for i in range(16) for j in range(16)]).reshape((16,16))





def euclidean_distance(s, m):

	'''Calculates the euclidean distance between a sample s and a mean template m
	Args:
		s (np.ndarray): Sample (nfeatures)
		m (np.ndarray): Template (nfeatures)
	Returns:
		(float) The Euclidean distance between s and m
	'''
	if m.shape == (16,16):
		m = m.reshape(256)
	return np.sqrt(np.sum((s - m)**2))
	





def euclidean_distance_classifier(X, X_mean):

	'''Classifies based on the euclidean distance between samples in X and template vectors in X_mean
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		X_mean (np.ndarray): Digits data (n_classes x nfeatures)
	Returns:
		(np.ndarray) predictions (nsamples)
	'''
	predictions = []
	for instance in X:
		distances = np.ndarray(10)
		for i in range(10):
			distances[i] = euclidean_distance(instance, X_mean[i])
		predictions.append(int(np.argmin(distances)))

	return np.array(predictions)





class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):

	"""Classify samples based on the distance from the mean feature value"""

	def __init__(self):

		self.X_mean_ = None


	def fit(self, X, y):

		"""
		This should fit classifier. All the "work" should be done here.
		Calculates self.X_mean_ based on the mean
		feature values in X for each class.
		self.X_mean_ becomes a numpy.ndarray of shape
		(n_classes, n_features)
		fit always returns self.
		"""

		self.X_mean_ = []
		classes = np.unique(y)
		for i in range(len(classes)):
			self.X_mean_.append(digit_mean(X, y, i))
		self.X_mean_ = np.array(self.X_mean_)
		return self



	def predict(self, X):

		"""
		Make predictions for X based on the
		euclidean distance from self.X_mean_
		"""

		min_distance_indexes = []
		for i in range(X.shape[0]):
			distances = []
			for j in range((self.X_mean_).shape[0]):
				distances.append(euclidean_distance(X[i],self.X_mean_[j]))
			min_distance_indexes.append(int(np.argmin(np.array(distances))))
		return np.array(min_distance_indexes)



	def score(self, X, y):

		"""
		Return accuracy score on the predictions
		for X based on ground truth y
		"""

		predictions = self.predict(X)
		accuracy = [1 if predictions[i] == int(y[i]) else 0 for i in range(predictions.shape[0])]
		acc = np.sum(accuracy)/len(accuracy)
		return acc


def evaluate_classifier(clf, X, y, folds=5):

	"""Returns the 5-fold accuracy for classifier clf on X and y
	Args:
		clf (sklearn.base.BaseEstimator): classifier
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
	Returns:
		(float): The 5-fold classification score (accuracy)
	"""

	scores = cross_val_score(clf, X, y, cv=KFold(n_splits=folds), scoring='accuracy')
	print("CV score = %f +-%f" % (np.mean(scores), np.std(scores)))
	


def calculate_priors(y, X):

	"""Return the a-priori probabilities for every class
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
	Returns:
		(np.ndarray): (n_classes) Prior probabilities for every class
	"""
	prior = np.array([np.where(y == i)[0].shape[0] for i in range(10)])/y.shape[0]
	return prior




class CustomNBClassifier(BaseEstimator, ClassifierMixin):

	"""Custom implementation Naive Bayes classifier"""

	def __init__(self, use_unit_variance=False):

		self.use_unit_variance = use_unit_variance
		self.digit_means = None
		self.digit_vars = None
		self.y = None

	def fit(self, X, y,flag=False):
		#if flag is true then all variances are equal to one

		"""
		This should fit classifier. All the "work" should be done here.
		Calculates self.X_mean_ based on the mean
		feature values in X for each class.
		self.X_mean_ becomes a numpy.ndarray of shape
		(n_classes, n_features)
		fit always returns self.
		"""
		self.y = y
		self.digit_means = np.ndarray(shape=(10,16,16)) #mean of each digit
		if flag:
			self.digit_vars = np.ones(shape=(10,16,16)) #var of each digit
		else:
			self.digit_vars = np.ndarray(shape=(10,16,16))
		for i in range(10):
			self.digit_means[i] = (digit_mean(X, y, i))
			if not(flag):
				self.digit_vars[i] = (digit_variance(X, y, i))+10**(-6)#add 10^(-6) because we divide these numbers as 0
		return self


	def predict(self, X):
		"""
		Make predictions for X based on the
		euclidean distance from self.X_mean_
		"""
		#naive bayes
		#P(yi|x1,x2,...,xn) = P(x1|yi)P(x2|yi)...P(xn|yi)*P(yi)
		#where p(yi|xi)~N(mu_i,sigma_i)
		#first calculate all dependent propabilities
		prior = calculate_priors(self.y,0)
		prop = np.ones(shape=(X.shape[0],10,16,16))
		for i in range(X.shape[0]):#number of sample
		    for j in range(10):#number of class
		                prop[i][j] = stats.norm.pdf(X[i].reshape(16,16),self.digit_means[j],self.digit_vars[j])#propability P(x_min|yi)

		#second calculate the naive bayes type
		mid_prop = np.ndarray(shape=(X.shape[0],10))
		for i in range(X.shape[0]):
		    for j in range(10):
		        mid_prop[i][j] = np.prod(prop[i][j], where = prop[i][j] > 0, keepdims = True)*prior[j]

		#finally keep the class with biggest propability
		total_pred = np.argmax(mid_prop,1)
		return total_pred



	def score(self, X, y):

		"""
		Return accuracy score on the predictions
		for X based on ground truth y
		"""
		predictions = self.predict(X)
		accuracy = [1 if predictions[i] == int(y[i]) else 0 for i in range(predictions.shape[0])]
		acc = np.sum(accuracy)/len(accuracy)
		return acc





def plot_clf(clf, X, y, labels):
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of Classifier')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    zeros = ax.scatter(
        X0[y == 0], X1[y == 0],
        c='blue', label=labels[0],
        s=60, alpha=0.9, edgecolors='k')
    ones = ax.scatter(
        X0[y == 1], X1[y == 1],
        c='red', label=labels[1], 
        s=60, alpha=0.9, edgecolors='k')
    twos = ax.scatter(
        X0[y == 2], X1[y == 2],
        c='yellow', label=labels[2], 
        s=60, alpha=0.9, edgecolors='k')
    threes = ax.scatter(
        X0[y == 3], X1[y == 3],
        c='white', label=labels[3], 
        s=60, alpha=0.9, edgecolors='k')
    fours = ax.scatter(
        X0[y == 4], X1[y == 4],
        c='black', label=labels[4], 
        s=60, alpha=0.9, edgecolors='k')
    fives = ax.scatter(
        X0[y == 5], X1[y == 5],
        c='orange', label=labels[5], 
        s=60, alpha=0.9, edgecolors='k')
    sixes = ax.scatter(
        X0[y == 6], X1[y == 6],
        c='green', label=labels[6], 
        s=60, alpha=0.9, edgecolors='k')
    sevens = ax.scatter(
        X0[y == 7], X1[y == 7],
        c='pink', label=labels[7], 
        s=60, alpha=0.9, edgecolors='k')
    eights = ax.scatter(
        X0[y == 8], X1[y == 8],
        c='gray', label=labels[8], 
        s=60, alpha=0.9, edgecolors='k')
    nines = ax.scatter(
        X0[y == 9], X1[y == 9],
        c='purple', label=labels[9], 
        s=60, alpha=0.9, edgecolors='k')
    
    ax.set_ylabel('Principal component 2')
    ax.set_xlabel('Principal component 1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()




def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt