from collections import OrderedDict
import numpy as np

from save_data import load_data

def dataset():
	arr_of_processed_tweets = load_data('data.dat')
	y_labels = [l[0] for l in arr_of_processed_tweets] #One row with all the labels
	# y_labels = np.reshape( y_labels , ( len(y_labels), ) )

	#now creating bag of words and calculating frequeny
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

	no_of_features = 10
	feature_labels = [ bagOfWords[i][0] for i in range(0,no_of_features)]
	# print(feature_labels)

	#Now we will make the X_test matrix contating frequency of feature label words

	X_test = np.zeros(shape = (len(userWords), no_of_features))

	i = 0
	for dictionary in userWords:
		for j in range(no_of_features):
			X_test[i][j] = dictionary.setdefault(feature_labels[j], 0)
		i = i + 1

	return X_test, y_labels

if __name__ == '__main__':

	X,y = dataset()
	print(X)
