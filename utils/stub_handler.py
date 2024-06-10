import pickle
import os

def save_stub(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_stub(path):
    with open(path, 'rb') as f:
        return pickle.load(f)