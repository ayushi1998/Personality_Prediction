import get_tweets
import text_process
from collections import OrderedDict
import numpy as np


def dataset():


	print("hI")

	tweets = get_tweets.get_tweets_list('../project/mbti_1.csv')
	# print(tweets)

	arr_of_processed_tweets = text_process(tweets)

	print(arr_of_processed_tweets)


	y_labels = [ l[0] for l in arr_of_processed_tweets] #One row with all the labels

	y_labels = np.reshape( y_labels , ( len(y_labels), 1 ) ) 


	#now creating bag of words and calculating frequeny 

	bagOfWords = {} #global dictinnary containing all the words with frequency for the whole data 
	userWords = [] #list of dictionaries which contains freuency of words of each user 

	for l in arr_of_processed_tweets:

		for list_of_words in l[1]:

			d = {} 

			for word in list_of_words:

				bagOfWords[word] = bagOfWords.setdefault(word, 0) + 1
				d[word] = d.setdefault(word, 0) + 1

			userWords.append(d)


	bagOfWords = OrderedDict(sorted(bagOfWords.items(), key=lambda(k,v):(v,k)))

	no_of_features = 1000

	feature_labels = list(bagOfWords.keys())[0:no_of_features]

	#Now we will make the X_test matrix contating frequency of feature label words

	X_test = np.zeros(shape = (len(userWords), no_of_features) )

	i=0
	for d in userWords :
		
		for j in range(no_of_features):

			X_test[i][j] = d.setdefault(feature_labels[j], 0)

		i = i +1

	return X_test, y_labels