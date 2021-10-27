import math

import numpy as np
from sklearn.preprocessing import OneHotEncoder

ATTRIBUTES = 13


# def oneHotEncoding(line):
#     O = np.zeros((ATTRIBUTES, ATTRIBUTES))
#     for i in line:
#


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidDerivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


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
    enc_x = OneHotEncoder(handle_unknown='ignore')
    enc_x.fit(x)
    X = enc_x.transform(x).toarray()
    enc_y = OneHotEncoder(handle_unknown='ignore')
    enc_y.fit(y)
    Y = enc_y.transform(y).toarray()
    return X, Y


def NeuralNetwork(x, y, MiniBatchSize, Features, Layers, Classes, lr):
    # output_layer = NotImplemented
    assert Features == x.shape[1], "Mismatch in number of features"
    theta = [None] + [0.01 * np.random.random((Layers[i], Layers[i + 1])) for i in range(len(Layers) - 1)] + [
        0.01 * np.random.random((Layers[-1], Classes))]
    theta[0] = np.random.random((x.shape[1] + 1, Layers[0]))
    m = x.shape[0]
    eps = float(1e-4)
    x = np.concatenate((np.ones((m, 1)), x), axis=1)
    layer_outputs = [None for i in range(len(Layers) + 2)]
    delta = [None for i in range(len(Layers) + 1)]
    iteration = 0
    prev_loss = math.inf
    for gg in range(50):
        # shuffle for stochastic gradient decent
        p = np.random.permutation(m)
        x, y = x[p], y[p]
        B = m // MiniBatchSize
        loss = 0
        for i in range(B):
            x_batch = x[i * MiniBatchSize: (i + 1) * MiniBatchSize]
            y_batch = y[i * MiniBatchSize: (i + 1) * MiniBatchSize]
            # do forward prop
            layer_outputs[0] = x_batch
            for j in range(len(Layers) + 1):
                # print(j, layer_outputs[j].shape)
                layer_outputs[j + 1] = sigmoid(layer_outputs[j] @ theta[j])
            output_layer = layer_outputs[-1]
            # print(layer_outputs)
            # break
            # calculate loss
            loss += np.sum((y_batch - output_layer) ** 2) / (2 * MiniBatchSize)
            # compute delta
            delta[-1] = (y_batch - output_layer).T * sigmoidDerivative(output_layer.T) / MiniBatchSize
            for k in range(len(Layers) - 1, -1, -1):
                delta[k] = (theta[k + 1] @ delta[k + 1]) * sigmoidDerivative(layer_outputs[k + 1].T)
            for t in range(len(theta)):
                theta[t] += lr * (layer_outputs[t].T @ delta[t].T)

        loss /= B
        print(abs(loss - prev_loss))
        # if abs(loss - prev_loss) < eps:
        #     break
        prev_loss = loss
        iteration += 1
        # print(it)
    # print(layer_outputs)
    return theta

def predict(x, y, theta):
    n = len(theta)
    pred = NotImplemented
    prev = NotImplemented
    for j in range(n):
        pred =


# def accuracy(y, y_pred):


def main():
    x, y = readData('poker-hand-training-true.data')
    theta = NeuralNetwork(x, y, 500, x.shape[1], [50, 100], 10, 0.1)
    print(output_layer)


main()
