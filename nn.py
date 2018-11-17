from sklearn.neural_network import MLPClassifier
from bag_of_words import get_bag_of_words
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from get_tweets import get_tweets_list
from sentiment_feature import get_sentiment

import numpy as np
import save_data

if __name__ == '__main__':
    X, y = get_bag_of_words('data.dat')
    # tweets = get_tweets_list('../project/mbti_1.csv')
    # sentiments,labels = get_sentiment(tweets)
    # save_data.save_data(sentiments,'sentiment_scores.dat')
    sentiments = save_data.load_data('sentiment_scores.dat')
    mean_sentiments = []
    var_sentiments = []
    for i in range(len(sentiments)):
        mean_sentiments.append(np.mean(sentiments[i]))
        var_sentiments.append(np.var(sentiments[i]))
    mean_sentiments = np.array(mean_sentiments)
    var_sentiments = np.array(var_sentiments)

    # print(X.shape)
    # print(mean_sentiments.shape)
    X = np.append(X,mean_sentiments.reshape(len(mean_sentiments),1),1)
    X = np.append(X,var_sentiments.reshape(len(var_sentiments),1),1)
    print(X.shape)
    #
    # y_all = [[label[i] for label in y] for i in range(4)]
    # classifiers = []
    # for i in range(4):
    #     clf = MLPClassifier(hidden_layer_sizes = (10,3,), max_iter = 500)
    #     X_train, X_test, y_train,y_test = train_test_split(X,y_all[i],test_size = 0.2)
    #
    #     clf.fit(X_train,y_train)
    #     test_pred = clf.predict(X_test)
    #     train_pred = clf.predict(X_train)
    #     print("Train Accuracy =", metrics.accuracy_score(y_train,train_pred))
    #     print("Test Accuracy =", metrics.accuracy_score(y_test,test_pred))
    #

    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2)
    classifier = MLPClassifier(hidden_layer_sizes = (50,),max_iter = 500)

    classifier.fit(X_train,y_train)
    test_pred = classifier.predict(X_test)
    train_pred = classifier.predict(X_train)
    print("Train Accuracy =", metrics.accuracy_score(y_train,train_pred))
    print("Test Accuracy =", metrics.accuracy_score(y_test,test_pred))

    conf = metrics.confusion_matrix(test_pred,y_test)
    plt.figure()
    sn.heatmap(conf,annot = True,fmt='.0f')
    plt.show()
