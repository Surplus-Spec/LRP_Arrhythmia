from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, GlobalMaxPooling1D
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
from load_data import get_data,shuffle_data
import config
import model_io

##################################################################
## model 구성하기
##
## 사용자 한 명씩 인증을 실시
dataset = "MITBIT"
path = "../data/MITBIT_Arrhythmia/"

if dataset == "MITBIT":
    mitbit_train = pd.read_csv(path + "mitbit_train.csv", header=None)
    mitbit_test = pd.read_csv(path + "mitbit_test.csv", header=None)

    mitbit_train, mitbit_test = np.array(mitbit_train), np.array(mitbit_test)

    train_X, train_Y = shuffle_data(mitbit_train[:,:-1],mitbit_train[:,-1])
    test_X, test_Y = shuffle_data(mitbit_test[:,:-1],mitbit_test[:,-1])

    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1, 1))
    train_Y = np_utils.to_categorical(train_Y, num_classes=len(config.label))

    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1, 1))
    test_Y = np_utils.to_categorical(test_Y, num_classes=len(config.label))

model = config.get_nn_model(train_X.shape[1])
model.train(train_X, train_Y, test_X, test_Y, batchsize=64, iters=15000, lrate=0.001, status = 1000)

model_io.write(model, '../Mitbit_NN_model.txt')

acc = np.mean(np.argmax(model.forward(test_X), axis=1) == np.argmax(test_Y, axis=1))
recall = recall_score(np.argmax(model.forward(test_X), axis=1),np.argmax(test_Y, axis=1),average = 'micro')
precision = precision_score(np.argmax(model.forward(test_X), axis=1),np.argmax(test_Y, axis=1),average = 'micro')

print(acc,recall,precision)