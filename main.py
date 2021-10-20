import numpy as np
from sklearn.preprocessing import OneHotEncoder

ATTRIBUTES = 13


# def oneHotEncoding(line):
#     O = np.zeros((ATTRIBUTES, ATTRIBUTES))
#     for i in line:
#


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def readData(filename):
    x = []
    y = []
    i = 0
    for line in open(filename, 'r'):
        temp = line.split(',' or '\n')
        _ = map(int, temp)
        # print(temp)
        z = list(_)
        # print(z)
        x.append(z[:-1])
        y.append(z[-1])
        i += 1
    x = np.array(x).reshape((len(x), -1))
    y = np.array(y).reshape((len(y), -1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(x)
    X = enc.transform(x).toarray()
    return X, y


def NeuralNetwork(x, y, MiniBatchSize, Features, Layers, Classes):
    assert Features == x.shape[1], "Mismatch in number of features"
    m = x.shape[0]
    x = np.concatenate((np.ones((m, 1)), x), axis=1)
    while True:
        # shuffle for stochastic gradient decent
        p = np.random.permutation(m)
        x, y = x[p], y[p]
        B = m // MiniBatchSize
        for i in range(B):
            x_batch = x[i * MiniBatchSize: (i + 1) * MiniBatchSize]
            y_batch = y[i * MiniBatchSize: (i + 1) * MiniBatchSize]
            # do forward prop
            # for i in range()


def main():
    x, y = readData('poker-hand-training-true.data')


main()
