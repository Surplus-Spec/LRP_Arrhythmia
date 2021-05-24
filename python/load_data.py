import sys
import scipy.io as sio
import os
import random
import numpy as np
import pandas as pd

BEAT_SIZE = 170 # sampling_rate

random.seed(153245)

def shuffle_data(x,y):
    s = np.arange(x.shape[0])
    random.shuffle(s)

    X = x[s]
    Y = y[s]

    end = int(x.shape[0] * 0.7)
    train_X = X[:end]
    train_Y = Y[:end]

    test_X = X[end:]
    test_Y = Y[end:]

    return train_X, train_Y

def get_data(x,y):
    s = np.arange(x.shape[0])
    random.shuffle(s)

    X = x[s]
    Y = y[s]

    end = x.shape[0] * 0.7
    train_X = X[:end]
    train_Y = Y[:end]

    test_X = X[end:]
    test_Y = Y[end:]

    return train_X, train_Y, test_X, test_Y

def RandomShuffle(Xt): # 데이터를 랜덤으로 섞는다.
    Xt = np.array(Xt)

    s = np.arange(Xt.shape[0])
    random.shuffle(s)
    Xt = Xt[s];

    return Xt