from sentiment_feature import get_sentiment
import numpy as np
from get_tweets import get_tweets_list
import save_data
import matplotlib.pyplot as plt
from bag_of_words import get_bag_of_words
if __name__ == '__main__':
    tweet_sentiment_scores = save_data.load_data('sentiment_scores.dat')
    X,y = get_bag_of_words('data.dat')
    # print(tweet_sentiment_scores)
    unique_labels = np.unique(y)

    means = np.zeros(unique_labels.shape)
    sample_mean = []
    for i in range(len(unique_labels)):
        label = unique_labels[i]

        label_scores = tweet_sentiment_scores[y == label]
        # print(label_scores)
        mean = np.mean(label_scores)
        variance = np.var(label_scores)
        sample_mean.append(np.mean(label_scores,axis = 0))
        position = unique_labels.searchsorted(label)
        means[i] = mean

    plt.figure()
    plt.scatter(unique_labels,means)

    plt.figure()
    for label,points in zip(unique_labels,sample_mean):
        plt.scatter([label] * len(points),points)
    plt.show()
