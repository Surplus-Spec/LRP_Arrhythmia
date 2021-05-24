'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 25.10.2016
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause

The purpose of this module is to demonstrate the process of obtaining pixel-wise explanations for given data points at hand of the MNIST hand written digit data set
with CNN models, using the LeNet-5 architecture.

The module first loads a pre-trained neural network model and the MNIST test set with labels and transforms the data such that each pixel value is within the range of [-1 1].
The data is then randomly permuted and for the first 10 samples due to the permuted order, a prediction is computed by the network, which is then as a next step explained
by attributing relevance values to each of the input pixels.

finally, the resulting heatmap is rendered as an image and (over)written out to disk and displayed.
'''

import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from keras.utils import np_utils
import importlib.util as imp
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np

from load_data import get_data,shuffle_data
import config
import model_io
import data_io
import render

#load a neural network, as well as the MNIST test data and some labels
nn = model_io.read('../Mitbit_CNN_model.txt') # 99.23% prediction accuracy
nn.drop_softmax_output_layer() #drop softnax output layer for analyses

dataset = "MITBIT"
path = "../data/MITBIT_Arrhythmia/"

if dataset == "MITBIT":
    mitbit_test = pd.read_csv(path + "mitbit_test.csv", header=None)

    mitbit_test = np.array(mitbit_test)

    test_X, test_Y = test_X, test_Y = mitbit_test[:,:-1],mitbit_test[:,-1]

    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1, 1))
    test_Y = np_utils.to_categorical(test_Y, num_classes=len(config.label))


X = test_X
Y = test_Y

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Y[:,0].astype(int)

acc = np.mean(np.argmax(nn.forward(X), axis=1) == np.argmax(Y, axis=1))
print('model test accuracy is: {:0.4f}'.format(acc))

#permute data order for demonstration. or not. your choice.
I = [np.arange(X.shape[0])]
#I = np.random.permutation(I)
get_label = set([])
k = 10
cn = 1
#predict and perform LRP for the 10 first samples
for i in I[0]:
    x = X[i:i+1,...]

    #forward pass and prediction
    ypred = nn.forward(x)

    if cn != np.argmax(Y[i]) or np.argmax(Y[i]) != np.argmax(ypred):
        continue
    print('True Class:     ', np.argmax(Y[i]))
    print('Predicted Class:', np.argmax(ypred),'\n')

    #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
    mask = np.zeros_like(ypred)
    mask[:,np.argmax(ypred)] = 1
    Rinit = ypred*mask

    R = nn.lrp(Rinit,'epsilon',1.)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140

    #sum over the third (color channel) axis. not necessary here, but for color images it would be.
    R = R.sum(axis=3)
    #same for input. create brightness image in [0,1].
    xs = ((x+1.)/2.).sum(axis=3)

    xs = np.reshape(xs,(xs.shape[1]))
    R = np.reshape(xs,(R.shape[1]))

    t = [ i for i in range(187) ]
    #render input and heatmap as rgb images
    #digit = render.ecg_to_rgb(xs, scaling = 3)
    #hm = render.hm_to_rgb(R, X = xs, scaling = 3, sigma = 2)
    #plt.imshow(R)
    for time,color in zip(t,R):
        if color > 0.7:
            plt.axvline(x=time,linewidth = 2.5,c = [color,0,0.1])
    plt.plot(t,xs,'k')

    #digit_hm = render.save_image([xs,R],'../heatmap.png')
    #data_io.write(R,'../heatmap.npy')

    #display the image as written to file
    #plt.imshow(digit_hm, interpolation = 'none')
    plt.axis('off')
    plt.savefig("../cnn_figure/class" + str(cn) + "/" + str(i) + ".png",dpi = 800)
    plt.show(block = False)



