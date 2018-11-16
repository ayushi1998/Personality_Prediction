from collections import OrderedDict
import numpy as np

from save_data import load_data

def get_bag_of_words(filename):
	arr_of_processed_tweets = load_data(filename)
	# arr_of_processed_tweets = []
	# for i in range(1,8000):
	# 	filename = "../data/%d.dat" % i
	# 	arr_of_processed_tweets.append(load_data(filename))

	"""
	y_labels: all labels extracted
	"""
	y_labels = [l[0] for l in arr_of_processed_tweets]

	"""
	now creating bag of words and calculating frequeny
	"""
	bagOfWords = {} #global dictionary containing frequency for all users
	userWords = [] #list of dictionary containing frequency for one user
	for l in arr_of_processed_tweets:
		d = {} #dictionary for a single user
		for list_of_words in l[1]:
			for word in list_of_words:
				bagOfWords[word] = bagOfWords.setdefault(word, 0) + 1
				d[word] = d.setdefault(word, 0) + 1
		userWords.append(d)

	bagOfWords = sorted(bagOfWords.items(), key=lambda kv: kv[1] , reverse = True )
	# print(bagOfWords)

	no_of_features = 1000
	feature_labels = [ bagOfWords[i][0] for i in range(0,no_of_features)]
	# print(feature_labels)

	"""
	X_test : contains frequency of feature label words
	"""
	X_test = np.zeros(shape = (len(userWords), no_of_features))

	i = 0
	for dictionary in userWords:
		for j in range(no_of_features):
			X_test[i][j] = dictionary.setdefault(feature_labels[j], 0)
		i = i + 1

	return X_test, y_labels

if __name__ == '__main__':

	X,y = get_bag_of_words('data.dat')
	print(X)
