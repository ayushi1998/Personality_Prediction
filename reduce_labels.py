from bag_of_words import get_bag_of_words
import numpy as np

def reduce_labels(y):
    y = np.array(y)
    unique_labels,counts = np.unique(y,return_counts = True)
    for l,c in zip(unique_labels,counts):
        if c<500:
            y[y==l] = 'NONE'
    return y

if __name__ == '__main__':
    X, y = get_bag_of_words('data.dat')

    y = reduce_labels(y)
    unique_labels,counts = np.unique(y,return_counts = True)
    print(unique_labels)
    print(counts)
