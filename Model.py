from bag_of_words import get_bag_of_words
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
def split():
    X , y = get_bag_of_words('data.dat')
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test 



def trainLR():
    X_train, X_test, y_train, y_test  = split()
    lr = LogisticRegression(solver='lbfgs')

    
    lr.fit(X_train, y_train)
    y_pred_class = lr.predict(X_test)
    #y_pred_prob = lr.predict_proba(X_test)[:, 1]

    print(lr.score(X_test, y_test))

    return lr


def trainSVM():
    X_train, X_test, y_train, y_test  = split()
    clf = SVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
 

    return clf


def RandomForest():
    X_train, X_test, y_train, y_test  = split()
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)

    Y_prediction = random_forest.predict(X_test)

    random_forest.score(X_train, y_train)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
    print(round(acc_random_forest,2,), "%")



    
    
if __name__ == '__main__':

	(RandomForest())
