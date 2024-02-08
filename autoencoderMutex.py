import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import KFold
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn import model_selection
import hyperopt
import random
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime
import logging
import datetime


class inputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, input_n_hidden1, input_n_hidden2, activation,is_train):
        super(inputSmallNetwork, self).__init__()
        self.l2 = None
        self.l1 = None
        self.n_hidden1 = input_n_hidden1
        self.n_hidden2 = input_n_hidden2
        self.is_train = is_train
        #self.concate = self.input_concate
        self.activation = activation
        self.is_train = False
        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        self.l1_layer = tf.keras.layers.Dense(self.n_hidden1, kernel_initializer='random_normal', name='layer1')
        self.l2_layer = tf.keras.layers.Dense(self.n_hidden2, kernel_initializer='random_normal', name='layer2')
        # self.l1=None
        # self.l2=None
        # self.layer2 = tf.keras.layers.Dense(self.n_hidden2, activation='relu')
        # tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))

    def call(self, inputs):
        l1 = self.l1_layer(inputs[0])
        l2 = self.l2_layer(inputs[1])
        l1 = tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))
        l2 = tf.keras.layers.BatchNormalization()(l2, training=bool(self.is_train))
        l1 = self.activation(l1)
        l2 = self.activation(l2)
        self.l1 = l1
        self.l2 = l2
        return self.l1, self.l2



class encoderNetwork(tf.keras.models.Model):
    def __init__(self, input_n_hidden1, input_n_hidden2, output_hidden, activation,is_train):
        super(encoderNetwork, self).__init__()
        # super().__init__(*args, **kwargs)
        self.l3 = None
        self.ensmallNetwork = inputSmallNetwork(input_n_hidden1, input_n_hidden2, activation,is_train)
        self.n_hidden3 = output_hidden
        self.l3_layer = tf.keras.layers.Dense(self.n_hidden3, kernel_initializer='random_normal', name='layer3')
        self.is_train = is_train
        # l3 = self.l3_layer(tf.concat([self.small_network.l1,self.small_network.l2 ], 1))
        # l1,l2 are bounded in l
        # print("-----layer3 shape", l3.shape)
        # l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))

        # self.concatenate = tf.keras.layers.Concatenate()
        # self.output_layer = tf.keras.layers.Dense(self.concate, activation='softmax')

    def call(self, inputs):
        output = self.ensmallNetwork(inputs)

        # self.l3_layer = tf.keras.layers.Dense(self.n_hiddensh, kernel_initializer=self.init, name='layer3')
        l3 = self.l3_layer(tf.concat([output[0], output[1]], 1))
        l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))
        self.l3 = l3
        return self.l3


class outputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, input_n_hidden1, input_n_hidden2, activation,is_train):
        super(outputSmallNetwork, self).__init__()
        self.l5 = None
        self.l6 = None
        self.n_hidden5 = input_n_hidden1
        self.n_hidden6 = input_n_hidden2
        self.activation = activation
        self.is_train = is_train
        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        self.l5_layer = tf.keras.layers.Dense(self.n_hidden5, kernel_initializer='random_normal', name='layer1')
        self.l6_layer = tf.keras.layers.Dense(self.n_hidden6, kernel_initializer='random_normal', name='layer2')
        # self.l1=None
        # self.l2=None
        # self.layer2 = tf.keras.layers.Dense(self.n_hidden2, activation='relu')
        # tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))

    def call(self, inputs):
        l5 = self.l5_layer(inputs[0])
        l6 = self.l6_layer(inputs[1])
        l5 = tf.keras.layers.BatchNormalization()(l5, training=bool(self.is_train))
        l6 = tf.keras.layers.BatchNormalization()(l6, training=bool(self.is_train))
        l5 = self.activation(l5)
        l6 = self.activation(l6)
        self.l5 = l5
        self.l6 = l6
        return self.l5, self.l6


class decoderNetwork(tf.keras.models.Model):
    def __init__(self, input_n_hidden1, input_n_hidden2,activation,is_train):
        super(decoderNetwork, self).__init__()

        self.l4_layer = tf.keras.layers.Dense(input_n_hidden1 + input_n_hidden2, kernel_initializer='random_normal',
                                              name='layer4')
        self.n_hidden5 = input_n_hidden1
        self.n_hidden6 = input_n_hidden2
        self.is_train = is_train
        #self.l5_layis_trainer = tf.keras.layers.Dense(input_n_hidden1, kernel_initializer=self.init, name='layer5')
        #self.l6_layer = tf.keras.layers.Dense(input_n_hidden2, kernel_initializer=self.init, name='layer6')

        self.outsmallNetwork=outputSmallNetwork(input_n_hidden1,input_n_hidden2,activation,is_train)

    def call(self, inputs):
        l4 = self.l4_layer(inputs)
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))
        self.l4 = l4

        output = tf.split(l4, [self.n_hidden5, self.n_hidden6], 1)
        l5,l6=self.outsmallNetwork(output)
        return l5,l6

# -------------------------------------------------------------------------

class Autoencoder(tf.keras.models.Model):
      def __init__(self, input_n_hidden1, input_n_hidden2, output_hidden, activation,is_train):
          super(Autoencoder, self).__init__()
          self.encoder = encoderNetwork(input_n_hidden1, input_n_hidden2, output_hidden, activation,is_train)
          self.decoder = decoderNetwork(input_n_hidden1, input_n_hidden2, activation,is_train)
          #self.inputData=X_train

      def call(self, inputs):
          encoded = self.encoder(inputs)
          decoded = self.decoder(encoded)
          return decoded
#--------------------------------------------------------------------------
if __name__ == '__main__':

    f = open(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_parameter_binary21.txt',
        'w+')
    f.close()

    fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_trials_binary_test21.pkl'
    with open(fname, 'wb+') as fpkl:
        pass
    # with open(fname, "wb") as f:
    # pass

    #start_time = time.perf_counter()
    selected_features = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\data\selected_features.csv',
        delimiter=',',
        skip_header=1)

    # log10 (fpkm + 1)
    inputhtseq = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\data\exp_intgr.csv',
        dtype=np.unicode_, delimiter=',', skip_header=1)
    inputhtseq = inputhtseq[:, 1:inputhtseq.shape[1]].astype(float)
    inputhtseq = np.divide((inputhtseq - np.mean(inputhtseq)), np.std(inputhtseq))
    print(inputhtseq.shape)

    # methylation β values
    inputmethy = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\data\mty_intgr.csv',
        dtype=np.unicode_, delimiter=',', skip_header=1)
    inputmethy = inputmethy[:, 1:inputmethy.shape[1]].astype(float)
    inputmethy = np.divide((inputmethy - np.mean(inputmethy)), np.std(inputmethy))
    print(inputmethy.shape)

    # tanh activation function
    act = tf.nn.tanh

    iterator=0
#------------------------------------------------------------------------------------------------------------
    #print('iteration', iterator)
    selected_feat_cmt = selected_features[np.where(selected_features[:, 0] == iterator + 1)[0], :]

    print('first source ...')
    htseq_cmt = selected_feat_cmt[np.where(selected_feat_cmt[:, 1] == 1)[0], :]
    htseq_nbr = len(htseq_cmt)
    htseq_sel_data = inputhtseq[:, htseq_cmt[:, 2].astype(int) - 1]

    print("second source ...")
    methy_cmt = selected_feat_cmt[np.where(selected_feat_cmt[:, 1] == 2)[0], :]
    methy_nbr = len(methy_cmt)
    methy_sel_data = inputmethy[:, methy_cmt[:, 2].astype(int) - 1]

    print("features size of the 1st dataset:", htseq_nbr)
    print("features size of the 2nd dataset:", methy_nbr)

    n_hidden1 = htseq_nbr
    print("shape of n_hidden1 ", n_hidden1)
    n_hidden2 = methy_nbr
    print("shape of n_hidden2 ", n_hidden2)
    output_hidden = 1
    if htseq_nbr > 1 and methy_nbr > 1:
        # split dataset to training and test data 80%/20%
        X_train1, X_test1 = model_selection.train_test_split(htseq_sel_data, test_size=0.2, random_state=1)
        X_train2, X_test2 = model_selection.train_test_split(methy_sel_data, test_size=0.2, random_state=1)
        print("The traning set of X1 train", X_train1.shape)
        print("The traning set of X2 train", X_train2.shape)
        sampleInput=[X_train1,X_train2]
        sae = Autoencoder(n_hidden1, n_hidden2, output_hidden, activation=act,is_train=False)
        resultMatrix=sae(sampleInput)
        #print(resultMatrix)
        print(resultMatrix[0].numpy().shape)
        print(resultMatrix[1].numpy().shape)

