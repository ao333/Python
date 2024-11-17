import pickle
import os

def DataDump(trees, filename, t):
    file = open(filename, 'wb')
    pickle.dump((t, trees), file)
    file.close()

def DataLoad(filename):
    file = open(filename, 'rb')
    (t, trees) = pickle.load(file)
    file.close()
    return (t, trees)
