from bag_of_words import get_bag_of_words
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from save_data import load_data
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

def split(X, y):

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test 


def LR(X_train, X_test, y_train, y_test):

    lr = LogisticRegression(solver='lbfgs', penalty = 'l2',  C = 2)
    lr.fit(X_train, y_train)
    y_pred_class = lr.predict(X_test)
    #y_pred_prob = lr.predict_proba(X_test)[:, 1]
    print(lr.score(X_test, y_test))

    return lr

def NBG(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))

def NBM(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))


def SVM(X_train, X_test, y_train, y_test):

    clf = SVC(gamma=0.001, C=1)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
 
    return clf


def KNN(X_train, X_test, y_train, y_test):

    knn = KNeighborsClassifier(n_neighbors=1000)
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_test))
    

def RandomForest(X_train, X_test, y_train, y_test):

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)

    Y_prediction = random_forest.predict(X_test)

    print(random_forest.score(X_test, y_test))


    
    
if __name__ == '__main__':

    """ Feature 1 - Bag of words """
    X, y = get_bag_of_words('data.dat') #Get processed data and labels from data folder

    """ Feature2 - Bag of words [20] + Sentiment + Words Per comment + Variance of words per comment """
    #X = load_data('featur2_data.dat')

    #X = np.append(X, , axis=1)

    y_EI = [l[0] for l in y] #for training for E vs I - stroes 1 bit label
    y_NS = [l[1] for l in y] #for training for N vs S - stroes 1 bit label
    y_FT = [l[2] for l in y] #for training for N vs T - stroes 1 bit label
    y_JP = [l[3] for l in y] #for training for J vs P - stroes 1 bit label

    y_category = [y_EI, y_NS, y_FT, y_JP, y]
    y_category_names = ['EvsI', 'NvsS', 'FvsT', 'JvsP', 'Overall']

    #XGbooster classifier for data correlation
    '''
    model = XGBClassifier()
    model.fit(X[:,:20], y)
    ax = plot_importance(model)
    fig = ax.figure
    fig.set_size_inches(100,100)
    pyplot.show()
    '''

    for i in range(5):    
        X_train, X_test, y_train, y_test = split(X, y_category[i])
        print(y_category_names[i])
        LR(X_train, X_test, y_train, y_test)
        
        
