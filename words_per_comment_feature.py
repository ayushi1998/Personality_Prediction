import get_tweets
import time
import numpy as np

def get_words_per_comment(tweets):

    feature = []
    labels = []

    for person in tweets:
        labels.append(person[0])
        list_of_tweets = person[1]
        word_count_for_user=[]
        for sentence in list_of_tweets:
            word_count = len(sentence.split())
            word_count_for_user.append(word_count)

        words_per_comment = sum(word_count_for_user)/len(word_count_for_user) #words per comment
        feature.append(words_per_comment)

    return np.reshape(feature,((-1,1))),np.array(labels)


def get_variance_of_words_per_comment(tweets):

    feature = []
    labels = []

    for person in tweets:
        labels.append(person[0])
        list_of_tweets = person[1]
        word_count_for_user=[]
        for sentence in list_of_tweets:
            word_count = len(sentence.split())
            word_count_for_user.append(word_count)

        variance_of_words_per_comment = np.var(word_count_for_user)
        feature.append(variance_of_words_per_comment )


    return np.reshape(feature,((-1,1))),np.array(labels)

def get_per_comment_features(tweets):

    feature = []
    labels = []

    for person in tweets:
        labels.append(person[0])
        list_of_tweets = person[1]
        word_count_for_user=[]
        for sentence in list_of_tweets:
            word_count = len(sentence.split())
            word_count_for_user.append(word_count)


        words_per_comment = sum(word_count_for_user)/len(word_count_for_user) #words per comment
        variance_of_words_per_comment = np.var(word_count_for_user)
        feature.append(words_per_comment)
        feature.append(variance_of_words_per_comment)

    return np.reshape(feature,((-1,2))),np.array(labels)


if __name__ == '__main__':
    
    start = time.time()
    
    tweets = get_tweets.get_tweets_list('../Project/mbti_1.csv')
    print(len(tweets))
    X, y = get_per_comment_features(tweets)
    end = time.time()
    print(X.shape,y)

    print("Time",end - start)

