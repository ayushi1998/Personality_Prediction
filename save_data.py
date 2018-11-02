import pickle

def save_data(data,filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_data(filename):
    with open(filename,"rb") as f:
        output = pickle.load(f)

    return output
