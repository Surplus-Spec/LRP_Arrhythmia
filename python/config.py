from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, GlobalMaxPooling1D
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import modules
import model_io

label = [i for i in range(5)]
dataset = "MITBIT" # PTB
input_size = 188 -1

nn_layer = 4
layer = [256, 512, 128, 32]

def get_cnn_model(Xtrain,Ytrain,Xtest,Ytest):
    maxnet = modules.Sequential([
        modules.Convolution(filtersize=(4, 1, 1, 2), stride=(1, 1)), \
        #modules.Rect(), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 2, 4), stride=(1, 1)), \
        #modules.Rect(), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 4, 8), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 8, 16), stride=(1, 1)), \
        #modules.Rect(), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 16, 8), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(2, 1, 8, len(label)), stride=(1, 1)), \
        modules.Flatten(), \
        modules.SoftMax()
    ])

    # train the network.
    maxnet.train(X=Xtrain, \
                 Y=Ytrain, \
                 Xval=Xtest, \
                 Yval=Ytest, \
                 iters=10000, \
                 lrate=0.001, \
                 batchsize=64)

    # save the network
    model_io.write(maxnet, '../Mitbit_CNN_model.txt')

    return maxnet

def get_cnn_model(Xtrain,Ytrain,Xtest,Ytest):
    maxnet = modules.Sequential([
        modules.Convolution(filtersize=(4, 1, 1, 2), stride=(1, 1)), \
        #modules.Rect(), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 2, 4), stride=(1, 1)), \
        #modules.Rect(), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 4, 8), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 8, 16), stride=(1, 1)), \
        #modules.Rect(), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 16, 8), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(2, 1, 8, len(label)), stride=(1, 1)), \
        modules.Flatten(), \
        modules.SoftMax()
    ])

    # train the network.
    maxnet.train(X=Xtrain, \
                 Y=Ytrain, \
                 Xval=Xtest, \
                 Yval=Ytest, \
                 iters=12000, \
                 lrate=0.001, \
                 batchsize=64)

    # save the network
    model_io.write(maxnet, '../Mitbit_CNN_model.txt')

    return maxnet

def get_nn_model(input_size):
    output_size = len(label)
    nn = modules.Sequential(
        [
            modules.Flatten(),
            modules.Linear(input_size, 256),
            modules.Linear(256, 512),
            modules.Linear(512, 128),
            modules.Linear(128, output_size),
            modules.SoftMax()
        ]
    )

    return nn