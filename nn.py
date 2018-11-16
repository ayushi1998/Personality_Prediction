from sklearn.neural_network import MLPClassifier
from bag_of_words import get_bag_of_words
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == '__main__':
    X, y = get_bag_of_words('data.dat')
    y_all = [[label[i] for label in y] for i in range(4)]
    classifiers = []
    for i in range(4):
        clf = MLPClassifier(hidden_layer_sizes = (20,20), max_iter = 500)
        X_train, X_test, y_train,y_test = train_test_split(X,y_all[i],test_size = 0.25)

        clf.fit(X_train,y_train)
        test_pred = clf.predict(X_test)
        train_pred = clf.predict(X_train)
        print("Train Accuracy =", metrics.accuracy_score(y_train,train_pred))
        print("Test Accuracy =", metrics.accuracy_score(y_test,test_pred))


    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.25)
    classifier = MLPClassifier(hidden_layer_sizes = (20,20,),max_iter = 500)

    classifier.fit(X_train,y_train)
    test_pred = classifier.predict(X_test)
    train_pred = classifier.predict(X_train)
    print("Train Accuracy =", metrics.accuracy_score(y_train,train_pred))
    print("Test Accuracy =", metrics.accuracy_score(y_test,test_pred))
