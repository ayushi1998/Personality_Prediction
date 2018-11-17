import get_tweets
from textblob import TextBlob
import time
import numpy as np

def get_sentiment(tweets):

    feature = []
    labels = []
    #list_of_tweets
    for person in tweets:
        labels.append(person[0])
        list_of_tweets = person[1]
        polarity_value=[]
        for sentence in list_of_tweets:
            test = TextBlob(sentence)
            polarity_value.append(test.sentiment.polarity)

        feature.append(polarity_value)



    return np.array(feature),np.array(labels)





if __name__ == '__main__':
    start = time.time()
    tweets = get_tweets.get_tweets_list('../project/mbti_1.csv')

    eq_len = []
    for tweet in tweets:
        if len(tweet[1])!=50:
            continue
        eq_len.append(tweet)

    tweets = eq_len
    print(len(eq_len))
    X, y = get_sentiment(tweets[0:20])
    end = time.time()
    # print(X)
    print("Time",end - start)
