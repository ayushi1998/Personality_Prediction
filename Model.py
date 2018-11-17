from bag_of_words import get_bag_of_words
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def split(X, y):

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test 


def LR(X_train, X_test, y_train, y_test):

    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_train, y_train)
    y_pred_class = lr.predict(X_test)
    #y_pred_prob = lr.predict_proba(X_test)[:, 1]
    print(lr.score(X_test, y_test))

    return lr


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

    random_forest.score(X_train, y_train)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
    print(round(acc_random_forest,2,), "%")



    
    
if __name__ == '__main__':

    X, y = get_bag_of_words('data.dat') #Get processed data and labels from data folder

    y_EI = [l[0] for l in y] #for training for E vs I - stroes 1 bit label
    y_NS = [l[1] for l in y] #for training for N vs S - stroes 1 bit label
    y_FT = [l[2] for l in y] #for training for N vs T - stroes 1 bit label
    y_JP = [l[3] for l in y] #for training for J vs P - stroes 1 bit label

    y_category = [y_EI, y_NS, y_FT, y_JP, y]
    y_category_names = ['EvsI', 'NvsS', 'FvsT', 'JvsP', 'Overall']

    for i in range(5):    
        X_train, X_test, y_train, y_test = split(X, y_category[i])
        print(y_category_names[i])
        RandomForest(X_train, X_test, y_train, y_test)
        
