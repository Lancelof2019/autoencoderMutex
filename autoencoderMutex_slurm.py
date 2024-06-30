import os

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
import copy
import multiprocessing
from functools import partial
import csv
from sklearn.preprocessing import MinMaxScaler
#import rpy2.robjects as ro
cancer_type="SKCM"
max_requirement=20
class inputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, n_hidden1, n_hidden2, activation, _init):
        super(inputSmallNetwork, self).__init__()
        self.l2 = None
        self.l1 = None
        self.n_input1 = n_hidden1
        self.n_input2 = n_hidden2
        # self.is_train = is_train
        # self.concate = self.input_concate
        self.activation = activation

        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        # self.l1_layer = tf.keras.layers.Dense(self.n_input1, kernel_initializer='random_normal', name='layer1')
        # self.l2_layer = tf.keras.layers.Dense(self.n_input2, kernel_initializer='random_normal', name='layer2')
        self.l1_layer = tf.keras.layers.Dense(self.n_input1, kernel_initializer=_init, name='layer1')
        self.l2_layer = tf.keras.layers.Dense(self.n_input2, kernel_initializer=_init, name='layer2')
        # self.l1=None
        # self.l2=None
        # self.layer2 = tf.keras.layers.Dense(self.n_hidden2, activation='relu')
        # tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))

    def call(self, inputs, is_train):
       # print("yes or no")
        #print("In the part of inputSmallNetwork",inputs[0])
        l1 = self.l1_layer(inputs[0])
       # print(type(inputs[0]))
       # print("l1 type is :",type(l1))
        l2 = self.l2_layer(inputs[1])
        #print("we good1?")
        self.is_train = is_train
        l1 = tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))
        l2 = tf.keras.layers.BatchNormalization()(l2, training=bool(self.is_train))
        l1 = self.activation(l1)
        l2 = self.activation(l2)

        self.l1 = l1
        self.l2 = l2
        # self.W1 = self.l1_layer.kernel
        # self.W2 = self.l2_layer.kernel
        return self.l1, self.l2

   # def get_weights(self):
       # if not self.l1_layer.built or not self.l2_layer.built:
           # raise ValueError("Weights have not been initialized yet.")
       # return self.l1_layer.kernel, self.l2_layer.kernel, self.l1_layer.bias, self.l2_layer.bias



class encoderNetwork(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, n_hiddensh, activation, _init):
        super(encoderNetwork, self).__init__()
        # super().__init__(*args, **kwargs)
        self.l3 = None
        self.ensmallNetwork = inputSmallNetwork(n_input1, n_input2, activation, _init)
        self.n_hidden3 = n_hiddensh
        # self.l3_layer = tf.keras.layers.Dense(self.n_hidden3, kernel_initializer='random_normal', name='layer3')
        self.l3_layer = tf.keras.layers.Dense(self.n_hidden3, kernel_initializer=_init, name='layer3')

        # l3 = self.l3_layer(tf.concat([self.small_network.l1,self.small_network.l2 ], 1))
        # l1,l2 are bounded in l
        # print("-----layer3 shape", l3.shape)
        # l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))

        # self.concatenate = tf.keras.layers.Concatenate()
        # self.output_layer = tf.keras.layers.Dense(self.concate, activation='softmax')

    def call(self, inputs, is_train):
        self.is_train = is_train
       # print("Ok or not")
        output = self.ensmallNetwork(inputs, self.is_train)
       # print("output is type :",type(output))
       # print("nice output")
        self.W1 = self.ensmallNetwork.l1_layer.kernel
        self.W2 = self.ensmallNetwork.l2_layer.kernel
        self.bias1 = self.ensmallNetwork.l1_layer.bias
        self.bias2 = self.ensmallNetwork.l2_layer.bias

        # self.l3_layer = tf.keras.layers.Dense(self.n_hiddensh, kernel_initializer=self.init, name='layer3')
        l3 = self.l3_layer(tf.concat([output[0], output[1]], 1))
        l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))

        self.l3 = l3
        self.Wsht = self.l3_layer.kernel
        self.bias3 = self.l3_layer.bias
        # self.Wsht = self.l3_layer.kernel
        return self.l3

   # def get_weights(self):
       # if not self.l3_layer.built:
           # raise ValueError("Weights have not been initialized yet.")
       # return self.l3_layer.kernel, self.l3_layer.bias


class outputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, n_input1, n_input2, activation, _init):
        super(outputSmallNetwork, self).__init__()
        self.l5 = None
        self.l6 = None
        self.n_hidden5 = n_input1
        self.n_hidden6 = n_input2
        self.activation = activation
        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        # self.l5_layer = tf.keras.layers.Dense(self.n_hidden5, kernel_initializer='random_normal', name='layer1')
        # self.l6_layer = tf.keras.layers.Dense(self.n_hidden6, kernel_initializer='random_normal', name='layer2')

        self.l5_layer = tf.keras.layers.Dense(self.n_hidden5, kernel_initializer=_init, name='layer5')
        self.l6_layer = tf.keras.layers.Dense(self.n_hidden6, kernel_initializer=_init, name='layer6')
        # self.l1=None
        # self.l2=None
        # self.layer2 = tf.keras.layers.Dense(self.n_hidden2, activation='relu')
        # tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))

    def call(self, inputs, is_train):
        #print("we good 2?")
        l5 = self.l5_layer(inputs[0])
        l6 = self.l6_layer(inputs[1])
        self.is_train = is_train
        l5 = tf.keras.layers.BatchNormalization()(l5, training=bool(self.is_train))
        l6 = tf.keras.layers.BatchNormalization()(l6, training=bool(self.is_train))
        l5 = self.activation(l5)
        #print("l5 is type of data:",type(l5))
        l6 = self.activation(l6)

        self.l5 = l5
        self.l6 = l6
        # self.W1t = self.l5_layer.kernel
        # self.W2t = self.l6_layer.kernel
        return self.l5, self.l6

   # def get_weights(self):
        #if not self.l6_layer.built or not self.l5_layer.built:
           #raise ValueError("Weights have not been initialized yet.")
       #return self.l5_layer.kernel, self.l6_layer.kernel, self.l5_layer.bias, self.l6_layer.bias


class decoderNetwork(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, n_hidden1, n_hidden2, activation, _init):
        super(decoderNetwork, self).__init__()

        # self.l4_layer = tf.keras.layers.Dense(n_hidden1 + n_hidden2, kernel_initializer='random_normal',
        # name='layer4')
        self.l4_layer = tf.keras.layers.Dense(n_hidden1 + n_hidden2, kernel_initializer=_init,
                                              name='layer4')
        # self.l5_layis_trainer = tf.keras.layers.Dense(input_n_hidden1, kernel_initializer=self.init, name='layer5')
        # self.l6_layer = tf.keras.layers.Dense(input_n_hidden2, kernel_initializer=self.init, name='layer6')
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden5 = n_input1
        self.n_hidden6 = n_input2
        self.outsmallNetwork = outputSmallNetwork(self.n_hidden5, self.n_hidden6, activation, _init)

    def call(self, inputs, is_train):
        l4 = self.l4_layer(inputs)
        self.is_train = is_train
        # self.is_train = True
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))

        self.l4 = l4
       # print("we good 3?" )
        # self.Wsh = self.l4_layer.kernel
        output = tf.split(l4, [self.n_hidden1, self.n_hidden2], 1)

        #print("output is type:",type(output))
        #print("we good 4?")
        l5, l6 = self.outsmallNetwork(output, self.is_train)

        self.Wsh = self.l4_layer.kernel
        self.bias4 = self.l4_layer.bias
        self.W1t = self.outsmallNetwork.l5_layer.kernel
        self.bias5 = self.outsmallNetwork.l5_layer.bias
        self.W2t = self.outsmallNetwork.l6_layer.kernel
        self.bias6 = self.outsmallNetwork.l6_layer.bias

        return l5, l6

    def get_weights(self):
        if not self.l4_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l4_layer.kernel, self.l4_layer.bias



class Autoencoder(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2,  iterator, activation):
        super(Autoencoder, self).__init__()
        self.bias6 = None
        self.bias5 = None
        self.W2t = None
        self.W1t = None
        self.bias4 = None
        self.Wsh = None
        self.Wsht = None
        self.bias3 = None
        self.bias2 = None
        self.bias1 = None
        self.W2 = None
        self.W1 = None
        self.n_hidden2 = None
        self.is_train = None
        self.n_hiddensh = 1
        self._init = None

        self.n_input1 = n_input1
        self.n_input2 = n_input2
        # self.n_hidden1 = n_hidden1
        # self.n_hidden2 = n_hidden2
        # self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.max_epochs = 500
        self.require_improvement = 20
        self.iterator = iterator
        self.activation = activation
        self.trigger = False
        # self.lamda = 0.13
        # self.alpha = 0.012
        # self.learning_rate = 0.032
        # self.inputData=X_train

    def call(self, inputs, is_train):
        self.sampleInput = inputs
        # print("self._init is:",self._init)

        # print("#################################################################")
        # print(inputs)
        # print("__________________________________________________")
        # print(self.sampleInput)
        # print("#################################################################")
        # self.temp_record = inputs
        # print("The first time of sampleInput",type(sampleInput))

        self.is_train = is_train
        encoded = self.encoder(inputs, self.is_train)
        decoded = self.decoder(encoded, self.is_train)

        # self.W1, self.W2,self.bias1,self.bias2= self.encoder.ensmallNetwork.get_weights()
        # self.W2 = self.encoder.ensmallNetwork.
        # self.Wsht,self.bias3= self.encoder.get_weights()
        # self.Wsh,self.bias4= self.decoder.get_weights()
        # self.W1t, self.W2t ,self.bias5,self.bias6= self.decoder.outsmallNetwork.get_weights()

        self.W1 = self.encoder.W1
        self.W2 = self.encoder.W2
        self.bias1 = self.encoder.bias1
        self.bias2 = self.encoder.bias2
        self.Wsht = self.encoder.Wsht
        self.bias3 = self.encoder.bias3

        self.Wsh = self.decoder.Wsh
        self.bias4 = self.decoder.bias4

        self.W1t = self.decoder.W1t
        self.bias5 = self.decoder.bias5
        self.W2t = self.decoder.W2t
        self.bias6 = self.decoder.bias6

        return decoded
        # self.W2t = self.decoder.outsmallNetwork.W2t
        # print(self.W1)
        #  print("======================================")
        # print(self.W2)
        # print("======================================")
        # print(self.Wsht)
        # print("======================================")
        # print(self.Wsh)
        # print("======================================")
        # print(self.W1t)
        # print("======================================")
        # print(self.W2t)
        # return decoded

    def L1regularization(self, weights):
        return tf.reduce_sum(tf.abs(weights))

    def L2regularization(self, weights, nbunits):
        return math.sqrt(nbunits) * tf.nn.l2_loss(weights)

    def lossfun(self, sampleInput, is_train, init,logger):
        ###########################################################################
        # print("self._init is:", init)
        #print("The length is :",sampleInput[0].shape)
        #print("The type of sampleInput[0]:",type(sampleInput[0]))

        if self.trigger == False:
            self.encoder = encoderNetwork(self.n_input1, self.n_input2, self.n_hiddensh, self.activation, init)
            self.decoder = decoderNetwork(self.n_input1, self.n_input2, self.n_hidden1, self.n_hidden2, self.activation,init)
            self.trigger = True
            # print("self._init is:", self._init)
        #logger.info(f"The self.trigger is :{self.trigger}")
        # print("#################################################################")
        # print(inputs)
        # print("__________________________________________________")
        # print(self.sampleInput)
        # print("#################################################################")
        # self.temp_record = inputs
        # print("The first time of sampleInput",type(sampleInput))
        self.is_train = is_train
        self.encoded = self.encoder(sampleInput, self.is_train)
        self.decoded = self.decoder(self.encoded, self.is_train)

        # self.W1, self.W2,self.bias1,self.bias2= self.encoder.ensmallNetwork.get_weights()
        # self.W2 = self.encoder.ensmallNetwork.
        # self.Wsht,self.bias3= self.encoder.get_weights()
        # self.Wsh,self.bias4= self.decoder.get_weights()
        # self.W1t, self.W2t ,self.bias5,self.bias6= self.decoder.outsmallNetwork.get_weights()

        self.W1 = self.encoder.W1
        self.W2 = self.encoder.W2
        self.bias1 = self.encoder.bias1
        self.bias2 = self.encoder.bias2
        self.Wsht = self.encoder.Wsht
        self.bias3 = self.encoder.bias3

        self.Wsh = self.decoder.Wsh
        self.bias4 = self.decoder.bias4

        self.W1t = self.decoder.W1t
        self.bias5 = self.decoder.bias5
        self.W2t = self.decoder.W2t
        self.bias6 = self.decoder.bias6
        # self.H = self.encodefun(X1, X2)
        # X1_, X2_ = self.decodefun(self.H)
        # self.get_weights()
        # print("The 2nd time of sampleInput", type(sampleInput))
        # self.compareOutPut = self.call(sampleInput,is_train)
        ###########################################################################
        self.compareOutPut = self.decoded
        # print("The 3rd time of sampleInput", type(sampleInput))
        sgroup_lasso = self.L2regularization(self.W1, self.n_input1 * self.n_hidden1) + \
                       self.L2regularization(self.W2, self.n_input2 * self.n_hidden2)
        # print(sgroup_lasso.shape)
        # lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("The big value is :", self.L2regularization(self.W1, self.n_input1 * self.n_hidden1))

        error = tf.reduce_mean(tf.square(sampleInput[0] - self.compareOutPut[0])) + tf.reduce_mean(
            tf.square(sampleInput[1] - self.compareOutPut[1]))
        cost = 0.5 * error + 0.5 * self.lamda * (1 - self.alpha) * sgroup_lasso + 0.5 * self.lamda * self.alpha * lasso

        #print("cost is :",cost)
        #print("we good 5?")
        # logger.info(f"The error is :{error}")
        # logger.info(f"The lasso is :{lasso}")
        # logger.info(f"The sgroup_lasso is :{sgroup_lasso}")
        self.l3=self.encoder.l3
        return cost



    # def train(self,inputs):
    def train(self, train_index, val_index, inputs, init,logger):
        # training data


        # train_input1 = inputs[0][train_index, :]  # 370 x 4/5
        # train_input2 = inputs[1][train_index, :]
        #
        # logger.info("---------------TRain starts---------")
        # # validation data
        # val_input1 = inputs[0][val_index, :]  # 370 x 1/5
        # val_input2 = inputs[1][val_index, :]

        train_input1 = tf.gather(inputs[0], train_index)
        train_input2 = tf.gather(inputs[1], train_index)

        logger.info("---------------TRain starts---------")

        val_input1 = tf.gather(inputs[0], val_index)
        val_input2 = tf.gather(inputs[1], val_index)


        # costs history:
        costs = []#long memoery
        costs_val = []#long memoery
        costs_val_inter = []
        costs_inter = []

        # for early stopping:
        best_cost = 0
        best_val_cost = 100000
        best_cost_cost = 100000
        stop = False
        last_improvement = 0
        # vn_samples = val_input1.shape[0]
        # n_samples = self.sampleInput[0].shape[0]
        # n_samples = train_input1.shape[0]  # size of the training set #370 x 4/5
        # vn_samples = val_input1.shape[0]  # size of the validation set#370 x 1/5
        epoch = 0
        counter = 0
        # self.batch_size=16
        # k = 5

        # print("train_input1 shape:", train_input1.shape)
        # print("train_input1 type:", type(train_input1))
        # print("train_input2 shape:", train_input2.shape)
        # print("train_input2 type:", type(train_input2))
        batch_xs1 = train_input1
        batch_xs2 = train_input2

        batch_vs1 = val_input1
        batch_vs2 = val_input2
        # print(batch_xs1)
        # print("batch_xs1 shape:", batch_xs1.shape)
        # # print("batch_xs1 type:", type(batch_xs1))
        # print("batch_xs2 shape:", type(batch_xs2))
        # print("batch_xs2 type:", type(batch_xs2))
        # print("batch_xs1 shape:", batch_xs1.shape)
        # print("batch_xs1 type:", type(batch_xs1))
        # print("batch_xs2 shape:", batch_xs2.shape)
        # print("batch_xs2 type:", type(batch_xs2))
        # print("------------------output-----------------------")
        # print(batch_xs1)
        # print(batch_xs2)
        # print("-----------------------------------------")
        # batch_xs1 = tf.convert_to_tensor(batch_xs1_const,dtype=tf.float32)
        # batch_xs2 = tf.convert_to_tensor(batch_xs2_const,dtype=tf.float32)

        #batch_xs1 = tf.Variable(batch_xs1_const, dtype=tf.float32)
        #batch_xs2 = tf.Variable(batch_xs2_const, dtype=tf.float32)
        # print("batch_xs1 shape:", batch_xs1.shape)
        # # print("batch_xs1 type:", type(batch_xs1))
        # print("batch_xs2 shape:", type(batch_xs1))
        while epoch < self.max_epochs and stop == False:
            avg_cost = 0.
            # Loop over all batches
            # for sample in mini_batches:





            self.is_train = True
            with tf.GradientTape(persistent=True) as tape:
                # 计算当前批次的损失
                current_loss = self.lossfun([batch_xs1, batch_xs2], self.is_train, init,logger)
                #print("ok current loss")
                # 计算梯度
            gradients = tape.gradient(current_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            cost = self.lossfun([batch_xs1, batch_xs2], self.is_train, init, logger)

            # avg_cost += cost[0] * len(sample) / n_samples
            avg_cost = cost
            #print("very good ,we made it ")
            ##########################################################################################################
            # seq = list(range(vn_samples))
            # mini_batches = [
            #     seq[k:k + self.batch_size]
            #     for k in range(0, vn_samples, self.batch_size)
            # ]
            #-----------------------------------------------------------------------------------------------------------
            avg_cost_val = 0.

            self.is_train = False
            # feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: False}
            cost_val = self.lossfun([batch_vs1, batch_vs2], self.is_train, init,logger)
            avg_cost_val = cost_val
            #print("very good ,we made it in validation")
            del tape
            # cost history since the last best cost
            costs_val_inter.append(avg_cost_val)
            costs_inter.append(avg_cost)
                # print("----------------------------cost_val:", cost_val,avg_cost_val, "-----------------------------")
            # cost history since the last best cost

            #logger.info("++++++++++++++++++++++++The validation trial ends-++++++++++++++++++++++++++++++++++")
            # logger.info("\n")
            costs_inter.append(avg_cost)#1.1041128635406494,1.1026578,1.08701980114183155
            costs_val_inter.append(avg_cost_val)#0.7548856141183044,0.7347194156375122,0.7152150869369507
            #costs_val_inter:1.0108890533447266
            #####################################################################################################

            if avg_cost_val < best_val_cost:
                # print("avg_cost_val is :",avg_cost_val)
                # save_sess = self.sess  # save session
                best_val_cost = avg_cost_val
                # print("###########################################")
                #  print("show me the costs_val_inter",costs_val_inter)
                # print("show me the costs_inter", costs_inter)
                logger.info(f"The value of cost in val :{best_val_cost}")
                best_cost = avg_cost
                logger.info(f"The value of cost in val :{best_cost}")
                costs_val += costs_val_inter  # costs history of the validation set
                costs += costs_inter  # costs history of the training set
                #print('-------------------------------------')
                #print(costs)
                #print('-------------------------------------')
                # print("show me the costs",costs)

                # print("show me the costs_val",costs_val)
                # print("###########################################")
                last_improvement = 0
                costs_val_inter = []
                costs_inter = []


            else:
                last_improvement += 1
                #print("Plus :"+ str(last_improvement))
                # costs_val += costs_val_inter  # costs history of the validation set
                # costs += costs_inter
                # costs_val_inter = []
                # costs_inter = []
                # print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~This is else part and we will see the value of avg_cost_val :{avg_cost_val} and best_val_cost:{best_val_cost}~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if last_improvement > self.require_improvement:
                # print("No improvement found during the (self.require_improvement) last iterations, stopping optimization.")
                # Break out from the loop.
                stop = True

                # self.sess = save_sess  # restore session with the best cost

            epoch += 1
            # print("++++++++++++++++++++++++++The end of while++++++++++++++++++++++++++++++++++++++")
            # self.histcosts = costs
            # self.histvalcosts = costs_val
        self.histcosts = ([float(tensor_histcosts.numpy()) for tensor_histcosts in costs])
        # self.histcosts = list(self.histcosts)
        # self.histcosts = [scaler_v1 for scaler_v1 in self.histcosts]
        self.histvalcosts = ([float(tensor_histvalcosts.numpy()) for tensor_histvalcosts in costs_val])

        #costs_total = self.lossfun(inputs, self.is_train, init, logger)

        #s_profile = self.l3.numpy().flatten()

        # 将数据写入 CSV 文件
       # filename = f'scoreProfile_{self.iterator}_{self.spliter}_{best_val_cost}_.csv'
       #  with open(filename, mode='w', newline='') as file:
       #      writer = csv.writer(file)
       #      for val in s_profile:
       #          writer.writerow([val])
        del train_input1
        del train_input2
        del batch_xs1
        del batch_xs2
        del inputs
        return best_cost, best_val_cost

    #################################################################################################
    def cross_validation(self, params, inputs,logger):
        # retrieve parameters
        #logger.info("------------------------------Cross validation starts-------------------------------------")
        self.batch_size = params['batch_size']
        self.n_hidden1 = params['units1']
        self.n_hidden2 = params['units2']
        self.alpha = params['alpha']
        self.lamda = params['lamda']
        self.learning_rate = params['learning_rate']
        self.cross_counter = 0
        # k fold validation
        k = 5
        # print("========Inputs=========")
        # print(type(inputs))
        self.require_improvement = 20
        # inputs=self.sampleInput
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(inputs)
        # print("__________________________________________________________________")
        # print(self.sampleInput)
        # self.inputdata=copy.deepcopy(inputs)
        # print(self.inputdata)
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        init = params['initializer']
        current_time = int(time.time())
        if init == 'normal':
            self._init = tf.keras.initializers.RandomNormal()
        if init == 'uniform':
            self._init = tf.keras.initializers.RandomUniform()
        if init == 'He':
            self._init = tf.keras.initializers.HeNormal()
        if init == 'xavier':
            self._init = tf.keras.initializers.GlorotNormal(current_time)

        opt = params['optimizer']
        if opt == 'SGD':
            # self.optimizer = tf.keras.optimizers.SGD()
            self.optimizer = tf.keras.optimizers.legacy.SGD()
        if opt == 'adam':
            # self.optimizer = tf.keras.optimizers.Adam()
            self.optimizer = tf.keras.optimizers.legacy.Adam()
        if opt == 'nadam':
            # self.optimizer = tf.keras.optimizers.Nadam()
            self.optimizer = tf.keras.optimizers.legacy.Nadam()
        if opt == 'Momentum':
            # self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate, momentum=0.9)
        if opt == 'RMSProp':
            # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            self.optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=self.learning_rate)

        # cross-validation
        data = np.concatenate([inputs[0], inputs[1]], axis=1)

        # print("data shape is ",data.shape)
        kf = KFold(n_splits=k, shuffle=True)  # k fold cross validation
        kf.get_n_splits(data)  # returns the number of splitting iterations in the cross-validator
        # data is component with 301+301
        # validation set loss
        loss_cv = 0
        val_loss_cv = 0
        min_loss_cv = 1000000
        min_val_loss_cv = 1000000
        self.spliter = 0
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        for train_index, val_index in kf.split(data):

            train_index_tensor = tf.convert_to_tensor(train_index, dtype=tf.int32)
            val_index_tensor = tf.convert_to_tensor(val_index, dtype=tf.int32)


            loss_cv, val_loss_cv = self.train(train_index_tensor, val_index_tensor, inputs_tensor, self._init,logger)  # self._init
            self.trigger = False
            loss_cv += loss_cv
            val_loss_cv += val_loss_cv
       # print("#############################finish one community data in iterator:"+str(self.iterator)+"########################")
        loss_cv = loss_cv / k
        val_loss_cv = val_loss_cv / k
        self.cross_counter = 0
        hist_costs = self.histcosts
        hist_val_costs = self.histvalcosts
       # print('-----------------------------------')
        #print(hist_costs)
       # print(hist_val_costs)
       # print('-----------------------------------')
       #  plt.figure()
       #  plt.plot(hist_costs)
       #  #logger.info(f"++++++++++++++++++++++++plot:{costs}+++++++")
       #  plt.ylabel('Loss')
       #  plt.xlabel('Iterations')
       #  plt.title("Learning rate =" + str(round(self.learning_rate, 9)))
       #  plt.savefig(
       #      r'C:\Users\gklizh\Documents\Workspace\code_and_data19\figure\loss_curve\train_picture_results_' + str(
       #          cancer_type) + "_" + str(self.iterator) + '_test07.png')
       #  plt.close()
        # model_params = {
        #     'W1': self.W1.numpy(),
        #     'W2': self.W2.numpy(),
        #     'bias1': self.bias1.numpy(),
        #     'bias2': self.bias2.numpy(),
        #     'Wsht': self.Wsht.numpy(),
        #     'bias3': self.bias3.numpy(),
        #     'Wsh': self.Wsh.numpy(),
        #     'bias4': self.bias4.numpy(),
        #     'W1t': self.W1t.numpy(),
        #     'bias5': self.bias5.numpy(),
        #     'W2t': self.W2t.numpy(),
        #     'bias6': self.bias6.numpy()
        # }
        # with open(r'C:\Users\gklizh\Documents\Workspace\code_and_data19\data\python_related\result\train_weights_biases_' + str(cancer_type) + '_' + str(self.iterator) + '.pkl', 'wb') as f:
        #     pickle.dump(model_params, f)
        return {'loss': val_loss_cv, 'status': STATUS_OK, 'params': params, 'loss_train': loss_cv,
                'history_loss': hist_costs, 'history_val_loss': hist_val_costs}

##############################################################################################test#######################
    def testprocess(self, inputs, iterator, params):
        #global _init, optimizer
        # self.batch_size  = params['batch_size']

        self.n_hidden1 = params['units1']
        self.n_hidden2 = params['units2']
        self.alpha = params['alpha']
        self.lamda = params['lamda']
        self.learning_rate = params['learning_rate']

        self.require_improvement = 50
        self.max_epochs = 2000

        init = params['initializer']
        if init == 'normal':
            self.init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
        if init == 'uniform':
            self.init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        current_time = int(time.time())
        if init == 'He':
            self.init = tf.keras.initializers.HeNormal(seed=current_time)
        if init == 'xavier':
            self.init = tf.keras.initializers.GlorotNormal(seed=current_time)
        self.init=self.init
        #self.init = _init
        opt = params['optimizer']

        if opt == 'SGD':
            # self.optimizer = tf.keras.optimizers.SGD()
            optimizer = tf.keras.optimizers.legacy.SGD()
        if opt == 'adam':
            # self.optimizer = tf.keras.optimizers.Adam()
            optimizer = tf.keras.optimizers.legacy.Adam()
        if opt == 'nadam':
            # self.optimizer = tf.keras.optimizers.Nadam()
            optimizer = tf.keras.optimizers.legacy.Nadam()
        if opt == 'Momentum':
            # self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate, momentum=0.9)
        if opt == 'RMSProp':
            # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=self.learning_rate)

        self.optimizer = optimizer
        # self._init = _init

        print("H.layer1:", self.n_hidden1, ", H.layer2:", self.n_hidden2)
        print( "lamda", self.lamda,  "alpha:", self.alpha, ", learning_rate:", self.learning_rate)
        print("initializer: ", init, ', optimizer:',  opt)

        # loss, res = self.test(inputs, iterator)
        costs = []  # long memoery
        costs_inter = []
        # for early stopping:
        best_cost = 10000
        #best_cost_cost = 100000
        stop = False
        last_improvement = 0
        epoch = 0
        counter = 0
        batch_xs1 = inputs[0]
        batch_xs2 = inputs[1]
        #print(batch_xs1)
        while epoch < self.max_epochs and stop == False:
            avg_cost = 0.
            # Loop over all batches
            # for sample in mini_batches:

            self.is_train = True
            with tf.GradientTape(persistent=True) as tape:

                current_loss = self.lossfun([batch_xs1, batch_xs2], self.is_train, self.init, logger=None)

            gradients = tape.gradient(current_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            cost = self.lossfun([batch_xs1, batch_xs2], self.is_train, self.init, logger=None)

            #avg_cost = cost

            costs_inter.append(cost)  #
            # print("-----------------------------The costs_inter is :----------------------")
            # print(costs_inter)
            # print("-----------------------------The costs_inter is :----------------------")
            #####################################################################################################

            if cost < best_cost:

                best_cost = cost
                costs += costs_inter  # costs history of the training set

                last_improvement = 0

                costs_inter = []

            else:
                last_improvement += 1

            if last_improvement > self.require_improvement:
                # print("No improvement found during the (self.require_improvement) last iterations, stopping optimization.")
                # Break out from the loop.
                stop = True

            epoch += 1
        cmt_scores = self.encoded
        cmt_scores=cmt_scores.numpy()
        #costs=costs.numpy()
        with open(
            r'../data/python_related/result/community/communityScores_compare_'+str(cancer_type)+"_"+str(iterator)+'_test10.csv',
            'w',newline='', encoding='utf-8') as csvfile:
             writer = csv.writer(csvfile, lineterminator='\n')
             [writer.writerow(r) for r in cmt_scores]
        print("--------------------------------------------------------------")
        print(costs)
        print("--------------------------------------------------------------")
        plt.figure()
        plt.plot(costs)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(round(self.learning_rate, 9)))
        plt.savefig(r'../figure/loss_curve/getscores_picture_' + str(cancer_type) + "_" + str(iterator) + '_test10.png')
        plt.close()

def partialProcess(iterator, selected_features, inputhtseq, inputmethy, act):
    trials = {}

    fname = r'../data/python_related/result/comm_trials_binary_' + str(
        cancer_type) + "_" + str(iterator) + '_test10.pkl'
    # with open(fname, 'wb+') as fpkl:
    #     pass
    print('iteration', iterator)
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

    n_hidden1 = htseq_nbr# the same as n_input1 below
    n_hidden2 = methy_nbr#

   ##################################################
    #create logger recorder

    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    log_filename = f'demoapp.{current_time}_test_{iterator}.log'


    logger = logging.getLogger(f'demoapp_{iterator}')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.info(f'Partial process started {iterator}')

    # scaler1 = MinMaxScaler()
    # scaler2 = MinMaxScaler()
    X_train1_data_orig = htseq_sel_data
    X_train2_data_orig = methy_sel_data
    # X_train1_data= tf.Variable(X_train1_data_orig, dtype=tf.float64)  # 指定数据类型为 float32
    # X_train2_data= tf.Variable(X_train2_data_orig, dtype=tf.float64)
    # 对X_train1和X_train2进行拟合和转换
   # X_train1_normalized = scaler1.fit_transform(X_train1_data)
    #X_train2_normalized = scaler2.fit_transform(X_train2_data)

    ##################################################

    if htseq_nbr > 1 and methy_nbr > 1:
        # split dataset to training and test data 80%/20%

        X_train1, X_test1 = model_selection.train_test_split(X_train1_data_orig, test_size=0.6, random_state=1)
        X_train2, X_test2 = model_selection.train_test_split(X_train2_data_orig, test_size=0.6, random_state=1)


        sampleInput = [X_train1, X_train2]#364x120
        #sampleInput = [X_train1_var, X_train2_var]
        #testInput = [X_test1, X_test2]
        print(type(X_train2))
        print(type(sampleInput))
        # print("===================In main :============================")
        # print(sampleInput)
        is_train = True
        #print(X_train1)
        n_input1 = X_train1.shape[1]
        n_input2 = X_train2.shape[1]
        print("data 1 shape of col:", n_input1)
        print("data 2 shape of col:", n_input2)
        sae = Autoencoder(n_input1, n_input2,  iterator, activation=act)
        # trainMatrix = sae(sampleInput, is_train)
        # print(type(sampleInput))
        # trials = Trials()
        # define the space of hyper parameters
        trial_label = f"trial_{iterator}"  #
        trials[trial_label] = Trials()
        space = {
            'units1': hp.choice('units1', range(1, n_hidden1)),
            'units2': hp.choice('units2', range(1, n_hidden2)),
            'batch_size': hp.choice('batch_size', [16, 8, 4]),
            'alpha': hp.choice('alpha', [0, hp.uniform('alpha2', 0, 1)]),
            'learning_rate': hp.loguniform('learning_rate', -5, -1),
            'lamda': hp.choice('lamda', [0, hp.loguniform('lamda2', -8, -1)]),
            'optimizer': hp.choice('optimizer', ["adam", "nadam", "SGD", "Momentum", "RMSProp"]),
            'initializer': hp.choice('initializer', ["xavier"]),
        }

        # train the HP optimization with 20 iterations
        cross_validation_with_input = partial(sae.cross_validation, inputs=sampleInput,logger=logger)


        best = fmin(cross_validation_with_input, space, algo=tpe.suggest, max_evals=15, trials=trials[trial_label])
        # best = fmin(sae.cross_validation, space, algo=tpe.suggest, max_evals=3, trials=trials[trial_label],args=(sampleInput,))
       # print(best)
        #print("finish the max evaluation in iterator:"+str(sae.iterator))
        with open(fname, "ab") as file:
            pickle.dump(trials[trial_label], file)

        # saeTest = Autoencoder(n_input1, n_input2, n_hidden1, n_hidden2, activation=act)
        # testMatrix = saeTest(sampleInput, is_train=True)

        #loss = train_test(params=hyperopt.space_eval(space, best), testInput=testInput, iterator=iterator,logger=logger)
        f = open(
            r'../data/python_related/result/comm_parameter_binary_' + str(
                cancer_type) + "_" + str(iterator) + '_test11.txt',
            'a+')
        # print("---------open the file to load the best items in best-----------------------------")

        for k, v in best.items():
            f.write(str(k) + ':' + str(v) + ',')
            # print("k",str(k))
            # print("v",str(v))
        # # print("The file f recording ends")
        # f.write("---------------------------------------------")
        # f.write(str(loss))
        # f.write('\n')
        # f.close()
        # logger.info(f'Partial process ended {iterator}')
        del htseq_sel_data
        del methy_sel_data
        # del trials
        del sae


# def parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num):
#    if num_processes is None or num_processes == 0:
#        num_processes = multiprocessing.cpu_count()
#    print("community_num:", community_num)
#    process_iteration = partial(partialProcess, selected_features=selected_features, inputhtseq=inputhtseq,
#                                inputmethy=inputmethy, act=act)

#    with multiprocessing.Pool(processes=num_processes) as pool:

#        # results = pool.map(partial(partialProcess, selected_features=selected_features, inputhtseq=inputhtseq, inputmethy=inputmethy), range(21))
#        pool.map(process_iteration, range(community_num))

""
def partialtestProcess(iterator, selected_features, inputhtseq, inputmethy, act):
    global inputs_tensor, best, n_input1, n_input2
    trials = {}

    fname = r'../data/python_related/result/comm_trials_binary_' + str(
        cancer_type) + "_" + str(iterator) + '_test10.pkl'

    # fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data19\data\python_related\result\comm_trials_binary_' + str(
    #     cancer_type) + "_" + str(iterator) + '_test10.pkl'
    # with open(fname, 'wb+') as fpkl:
    #     pass
    print('iteration', iterator)
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

    # n_hidden1 = htseq_nbr# the same as n_input1 below
    # n_hidden2 = methy_nbr#

    ##################################################
    # create logger recorder

    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    log_filename = f'demoapp.{current_time}_test_{iterator}.log'

    logger = logging.getLogger(f'demoapp_{iterator}')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.info(f'Partial process started {iterator}')
    input_data = open(fname, 'rb')

    ##################################################

    if htseq_nbr > 1 and methy_nbr > 1:
        # split dataset to training and test data 80%/20%
        trials = pickle.load(input_data)
        best = trials.best_trial['result']['params']
        X_train1 = htseq_sel_data
        X_train2 = methy_sel_data


        sampleInput = [X_train1, X_train2]  # 364x120

        is_train = True
        # print(X_train1)
        n_input1 = X_train1.shape[1]
        n_input2 = X_train2.shape[1]
        print("data 1 shape of col:", n_input1)
        print("data 2 shape of col:", n_input2)

        inputs_tensor = tf.convert_to_tensor(sampleInput, dtype=tf.float32)
        sae = Autoencoder(n_input1, n_input2, iterator, activation=act)

        # space = {
        #    'units1': hp.choice('units1', range(1, n_hidden1)),
        #    'units2': hp.choice('units2', range(1, n_hidden2)),
        #    'batch_size': hp.choice('batch_size', [16, 8, 4]),
        #    'alpha': hp.choice('alpha', [0, hp.uniform('alpha2', 0, 1)]),
        #    'learning_rate': hp.loguniform('learning_rate', -5, -1),
        #    'lamda': hp.choice('lamda', [0, hp.loguniform('lamda2', -8, -1)]),
        #    'optimizer': hp.choice('optimizer', ["adam", "nadam", "SGD", "Momentum", "RMSProp"]),
        #    'initializer': hp.choice('initializer', ["xavier"]),
        # }
        sae.testprocess(inputs_tensor, iterator, best)

        del htseq_sel_data
        del methy_sel_data
        # del trials
        del sae



# def parallel_testprocessing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num):
#    if num_processes is None or num_processes == 0:
#        num_processes = multiprocessing.cpu_count()
#    print("community_num:", community_num)
#    process_testiteration = partial(partialtestProcess, selected_features=selected_features, inputhtseq=inputhtseq,
#                                inputmethy=inputmethy, act=act)

#    with multiprocessing.Pool(processes=num_processes) as pool:

#        # results = pool.map(partial(partialProcess, selected_features=selected_features, inputhtseq=inputhtseq, inputmethy=inputmethy), range(21))
#        pool.map(process_testiteration, range(community_num))



if __name__ == '__main__':
    #f = open(
       # r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_parameter_binary26.txt',
       # 'w+')
    #f.close()

    #fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_trials_binary_test26.pkl'
   # with open(fname, 'wb+') as fpkl:
    #    pass


    community_index = int(os.getenv('SLURM_ARRAY_TASK_ID', '1')) - 1
    
    feature_path = os.path.join(r'/projappl/project_2010541/data/python_related/data',
                         cancer_type + "_selected_features01.csv")

    selected_features = np.genfromtxt(feature_path,delimiter=',', skip_header=1)
    # log10 (fpkm + 1)

    exp_path = os.path.join(r'/projappl/project_2010541/data/python_related/data',
                            cancer_type + "_exp_intgr01.csv")
    inputhtseq = np.genfromtxt(exp_path,dtype=np.unicode_, delimiter=',', skip_header=1)
    inputhtseq = inputhtseq[:, 1:inputhtseq.shape[1]].astype(float)
    inputhtseq = np.divide((inputhtseq - np.mean(inputhtseq)), np.std(inputhtseq))
    print(inputhtseq.shape)
    mty_path = os.path.join(r'/projappl/project_2010541/data/python_related/data',
                            cancer_type + "_mty_intgr01.csv")
    # methylation β values
    inputmethy = np.genfromtxt(mty_path, dtype=np.unicode_, delimiter=',', skip_header=1)
    inputmethy = inputmethy[:, 1:inputmethy.shape[1]].astype(float)
    inputmethy = np.divide((inputmethy - np.mean(inputmethy)), np.std(inputmethy))
    print(inputmethy.shape)
    #community_num = 21

    act = tf.nn.tanh
    # C:\Program Files\R\R-4.3.3\
    # 使用R的print函数来打印一个消息

    #ro.r('options(stringsAsFactors = F)')
    #ro.r('suppressPackageStartupMessages(library(NbClust))')
    #ro.r('suppressPackageStartupMessages(library(ggplot2))')
    #ro.r('suppressPackageStartupMessages(library(grid))')
    #ro.r('suppressPackageStartupMessages(library(ComplexHeatmap))')
    #ro.r('suppressPackageStartupMessages(library(circlize))')
    #ro.r('suppressPackageStartupMessages(library(tidyverse))')
    #ro.r('suppressPackageStartupMessages(library(maftools))')
    #load_file = 'C:/Users/gklizh/Documents/Workspace/code_and_data12/data/spinglass/melanet_cmt.RData'

    # 将文件路径传递给R环境
    #ro.r(f"load_file <- '{load_file}'")

    # 使用这个变量来读取RDS文件
    #ro.r("melanet_cmt <- readRDS(load_file)")

    # 获取社区的数量
   # ro.r('number_of_communities <- length(melanet_cmt)')

    # 在Python中获取值
    #community_num = ro.r('number_of_communities')[0]
    #print(community_num)


    #num_processes=7
    #num_processes = 8
    #community_num = 25
    # parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num)
    #parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num)

    #parallel_testprocessing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num)
    partialProcess(community_index, selected_features, inputhtseq, inputmethy, act)
    
    
    partialtestProcess(community_index, selected_features, inputhtseq, inputmethy, act)
    # with open('script.py', 'r') as file:
    #     script_content = file.read()
    #
    # # 执行读取的内容
    # exec(script_content)
    # tanh activation function

    # trials = {}
    # run the autoencoder for communities

# for trial_label, trial in trials.items():
# print(f"\nData for {trial_label}:")
# for trial_result in trial.trials:
# print(trial_result)
