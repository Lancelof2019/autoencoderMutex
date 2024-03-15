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
from sklearn.preprocessing import MinMaxScaler
import rpy2.robjects as ro


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
        l1 = self.l1_layer(inputs[0])
        l2 = self.l2_layer(inputs[1])
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
        output = self.ensmallNetwork(inputs, self.is_train)

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
        l5 = self.l5_layer(inputs[0])
        l6 = self.l6_layer(inputs[1])
        self.is_train = is_train
        l5 = tf.keras.layers.BatchNormalization()(l5, training=bool(self.is_train))
        l6 = tf.keras.layers.BatchNormalization()(l6, training=bool(self.is_train))
        l5 = self.activation(l5)
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
        # self.Wsh = self.l4_layer.kernel
        output = tf.split(l4, [self.n_hidden1, self.n_hidden2], 1)
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
    def __init__(self, n_input1, n_input2, n_hidden1, n_hidden2, iterator, activation):
        super(Autoencoder, self).__init__()
        self.is_train = None
        self.n_hiddensh = 1
        self._init = None

        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        # self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.max_epochs = 1000
        self.require_improvement = 1000
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

        if self.trigger == False:
            self.encoder = encoderNetwork(self.n_input1, self.n_input2, self.n_hiddensh, self.activation, init)
            self.decoder = decoderNetwork(self.n_input1, self.n_input2, self.n_hidden1, self.n_hidden2, self.activation,
                                          init)
            self.trigger = True
            # print("self._init is:", self._init)
        logger.info(f"The self.trigger is :{self.trigger}")
        # print("#################################################################")
        # print(inputs)
        # print("__________________________________________________")
        # print(self.sampleInput)
        # print("#################################################################")
        # self.temp_record = inputs
        # print("The first time of sampleInput",type(sampleInput))
        self.is_train = is_train
        encoded = self.encoder(sampleInput, self.is_train)
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
        # self.H = self.encodefun(X1, X2)
        # X1_, X2_ = self.decodefun(self.H)
        # self.get_weights()
        # print("The 2nd time of sampleInput", type(sampleInput))
        # self.compareOutPut = self.call(sampleInput,is_train)
        ###########################################################################
        self.compareOutPut = decoded
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

        logger.info(f"The error is :{error}")
        logger.info(f"The lasso is :{lasso}")
        logger.info(f"The sgroup_lasso is :{sgroup_lasso}")
        return cost

    # def optiGradient(self):
    # def train_step(self,batch_xs1, batch_xs2):
    # is_train=True
    # with tf.GradientTape() as tape:
    #   current_loss = self.lossfun(batch_xs1, batch_xs2)
    # gradients = tape.gradient(current_loss, self.trainable_variables)
    # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    # return current_loss

    # def train(self,inputs):
    def train(self, train_index, val_index, inputs, init,logger):
        # training data

        # inputs = [self.inputdata[0].numpy(),self.inputdata[1].numpy()]

        train_input1 = inputs[0][train_index, :]  # 370 x 4/5
        train_input2 = inputs[1][train_index, :]

        logger.info("---------------TRain starts---------")
        # validation data
        val_input1 = inputs[0][val_index, :]  # 370 x 1/5
        val_input2 = inputs[1][val_index, :]

        # costs history:
        costs = []
        costs_val = []
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
        n_samples = train_input1.shape[0]  # size of the training set #370 x 4/5
        vn_samples = val_input1.shape[0]  # size of the validation set#370 x 1/5
        epoch = 0
        counter = 0
        # self.batch_size=16
        # k = 5

     #
        while epoch < self.max_epochs and stop == False:
            # for(self.max)
            # train the model on the training set by mini batches
            # shuffle then split the training set to mini-batches of size self.batch_size
            # logger.info(
            # f"#################################################epoch :{epoch}#################################################")
            seq = list(range(n_samples))  # 370 x 4/5
            # print("The number of n_samples",n_samples)#370 x 4/5
            random.shuffle(seq)
            mini_batches = [
                seq[k:k + self.batch_size]
                for k in range(0, n_samples, self.batch_size)
            ]

            avg_cost = 0.  # the average cost of mini_batches
            avg_cost_val = 0.
            # print(self.sampleInput[0].shape)
            # print(self.sampleInput[1].shape)

            logger.info(
                "----------------------one trial for train samples starts one epoch ------------------------")
            for sample in mini_batches:
                # print("#############Sample:", len(sample))
                # s1 = self.sampleInput[0].numpy()
                # s2 = self.sampleInput[1].numpy()
                # print(s1)
                # print(s2)
                # print(type(inputs))
                # print(type(inputs[0]))
                # batch_xs1 = inputs[0][sample][:]
                # batch_xs2 = inputs[1][sample][:]
                batch_xs1 = train_input1[sample][:]
                batch_xs2 = train_input2[sample][:]

                # batch_xs1 = self.sampleInput[0][sample][:]
                # batch_xs2 = self.sampleInput[1].numpy()[sample][:]

                self.is_train = True
                # loss = self.train_step(batch_xs1, batch_xs2)
                # feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: True}
                # cost = self.sess.run([self.loss_, self.train_step], feed_dict=feed_dictio)
                # avg_cost += cost[0] * len(sample) / n_samples

                with tf.GradientTape() as tape:
                    # 计算当前批次的损失
                    # if epoch == 0 and counter == 0:
                    # current_loss = self.loss(batch_xs1, batch_xs2)
                    # else:
                    current_loss = self.lossfun([batch_xs1, batch_xs2], self.is_train, init,logger)  # 1

                #logger.info(self.trainable_variables)

                # 计算梯度
                gradients = tape.gradient(current_loss, self.trainable_variables)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # print(gradients)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # 更新模型的参数
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                logger.info(
                    f"----------------------check if the weighths has been approved in this patch -----------------------------")
                # logger.info(self.trainable_variables)
                logger.info(
                    "-----------------------------------------patch ends in train-----------------------------------------")
                cost = self.lossfun([batch_xs1, batch_xs2], self.is_train, init,logger)  # 2
                logger.info(f"The value of cost in train cost:{cost}")
                # if cost==
                # cost = current_loss
                # print("---------------------train costs-------------")
                # print("cost:",cost)

                # print("-----------------------Kosten---------------------------",cost)
                # print("---------------------------------------------")
                avg_cost += cost * len(sample) / n_samples#go up
                counter += 1
                logger.info(f"The value of cost in train cost:{avg_cost}")
                # print("----------------------------cost_train:", cost, avg_cost, "-----------------------------")
            # print("---------------------train costs ends-------------")
            ##########################################################################################################
            seq = list(range(vn_samples))
            mini_batches = [
                seq[k:k + self.batch_size]
                for k in range(0, vn_samples, self.batch_size)
            ]
            counter = 0
            logger.info("--------------------------The validation trial starts------------------------------------")
            for sample in mini_batches:
                batch_xs1 = val_input1[sample][:]
                batch_xs2 = val_input2[sample][:]
                self.is_train = False
                # print("#############Sample:",len(sample))
                # feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: False}
                cost_val = self.lossfun([batch_xs1, batch_xs2], self.is_train, init,logger)
                logger.info(f"The value of cost_val in train cost:{cost_val}")
                #logger.info(self.trainable_variables)
                # self.lossfun(batch_xs1, batch_xs2)
                # print("---------------cost[0] val territory")

                # print(cost_val)
                # print("---------------cost[0] val territory ends")
                avg_cost_val += cost_val * len(sample) / vn_samples
                logger.info(f"The value of val in train cost:{avg_cost_val}")
                # print("----------------------------cost_val:", cost_val,avg_cost_val, "-----------------------------")
            # cost history since the last best cost

            logger.info("++++++++++++++++++++++++The validation trial ends-++++++++++++++++++++++++++++++++++")
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

                # print("show me the costs",costs)

                # print("show me the costs_val",costs_val)
                # print("###########################################")
                last_improvement = 0
                costs_val_inter = []
                costs_inter = []
                logger.info(
                    f"++++++++++++++++++++++++costs list {costs} -++++++++++++++++++++++++++++++++++")

            else:
                last_improvement += 1
                # costs_val += costs_val_inter  # costs history of the validation set
                # costs += costs_inter
                # costs_val_inter = []
                # costs_inter = []
                # print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~This is else part and we will see the value of avg_cost_val :{avg_cost_val} and best_val_cost:{best_val_cost}~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if last_improvement > 99:
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
            logger.info(f"++++++++++++++++++++++++The batches {self.iterator} ends-++++++++++++++++++++++++++++++++++")
            logger.info(
                f"++++++++++++++++++++++++The batches last_improvement:{last_improvement}++++++++++++++++++++++++++++++++++")
            return best_cost, best_val_cost

    #################################################################################################
    def cross_validation(self, params, inputs,logger):
        # retrieve parameters
        logger.info("------------------------------Cross validation starts-------------------------------------")
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
        self.max_epochs = 1000
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
        # counter=0
        # print("train check self.trainable_variables()",self.trainable_variables())
        self.spliter = 0
        for train_index, val_index in kf.split(data):
            # counter+=1
            # logger.info(f"train_index:{train_index.shape}")
            # logger.info(f"val_index:{val_index.shape}")

            # train the model
            # logger.info(
            # "-----------------------One iteration in for train_index, val_index in kf.split(data)-----------------------------")

            loss_cv, val_loss_cv = self.train(train_index, val_index, inputs, self._init,logger)  # self._init
            self.spliter += 1
            # self.cross_counter += 1
            # best_train,best_val
            loss_cv += loss_cv
            val_loss_cv += val_loss_cv

        loss_cv = loss_cv / k
        val_loss_cv = val_loss_cv / k
        self.cross_counter = 0
        hist_costs = self.histcosts
        hist_val_costs = self.histvalcosts
        logger.info("************************The crossvalidation ->hiscosts*********************")
        logger.info(hist_costs)
        logger.info("************************The crossvalidation ->hist_val_costs*********************")
        logger.info(hist_val_costs)

        # self.sess.close()
        # tf.reset_default_graph()
        # del self.sess
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%trian%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(self.histcosts)
        # print(self.histvalcosts)
        # print(val_loss_cv)
        # print(STATUS_OK)
        # print(loss_cv)
        # print(hist_costs)
        # print(hist_val_costs)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%trian%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info("--------------------------Cross validation ends-----------------------------------------")
        return {'loss': val_loss_cv, 'status': STATUS_OK, 'params': params, 'loss_train': loss_cv,
                'history_loss': hist_costs, 'history_val_loss': hist_val_costs}


def train_test(params, testInput, iterator,logger):
    global _init
    logger.info("------------test train starts--------------------")

    n_input1 = testInput[0].shape[1]
    n_input2 = testInput[1].shape[1]
    batch_size = params['batch_size']
    n_hidden1 = params['units1']

    n_hidden2 = params['units2']

    alpha = params['alpha']
    lamda = params['lamda']
    learning_rate = params['learning_rate']
    act = tf.nn.tanh
    init = params['initializer']
    current_time = int(time.time())
    if init == 'normal':
        _init = tf.keras.initializers.RandomNormal()
    if init == 'uniform':
        _init = tf.keras.initializers.RandomUniform()
    if init == 'He':
        _init = tf.keras.initializers.HeNormal()
    if init == 'xavier':
        _init = tf.keras.initializers.GlorotNormal(current_time)

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
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate, momentum=0.9)
    if opt == 'RMSProp':
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate)
    # test_input1 = self.test_data1
    # test_input2 = self.test_data2

    test_input1 = testInput[0]
    test_input2 = testInput[1]
    # n_input1 = testInput[0].shape[1]
    # n_input2 = testInput[1].shape[1]
    # saeTest = Autoencoder(n_input1, n_input2, n_hidden1, n_hidden2, activation=act)

    sae_test = Autoencoder(n_input1, n_input2, n_hidden1, n_hidden2, iterator, activation=act)
    sae_test.optimizer = optimizer
    sae_test.alpha = alpha
    sae_test.lamda = lamda
    sae_test.learning_rate = learning_rate
    sae_test.is_train = True

    costs = []
    costs_inter = []

    epoch = 0
    counter = 0
    best_cost = 100000
    stop = False
    n_samples = test_input1.shape[0]
    last_improvement = 0
    logger.info("++++++++++++++++++++++++test_train-++++++++++++++++++++++++++++++++++")
    while epoch < sae_test.max_epochs and stop == False:
        # train the model on the test set by mini batches
        # shuffle then split the test set into mini-batches of size self.batch_size
        seq = list(range(n_samples))
        random.shuffle(seq)
        mini_batches = [
            seq[k:k + batch_size]
            for k in range(0, n_samples, batch_size)
        ]
        avg_cost = 0.

        # Loop over all batches
        for sample in mini_batches:
            batch_xs1 = test_input1[sample][:]
            batch_xs2 = test_input2[sample][:]
            is_train = True
            # logger.info(batch_xs1.shape)
            # logger.info(batch_xs2.shape)
            with tf.GradientTape() as tape:
                # 计算当前批次的损失
                # current_loss = self.lossfun(batch_xs1, batch_xs2)
                # if epoch == 0 and counter == 0:
                # current_loss = self.loss(batch_xs1, batch_xs2)
                # else:
                # current_loss = self.lossfun(batch_xs1, batch_xs2)
                current_loss = sae_test.lossfun([batch_xs1, batch_xs2], is_train, _init,logger)
                # 计算梯度
            gradients = tape.gradient(current_loss, sae_test.trainable_variables)
            # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
            # print(gradients)
            # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
            # 更新模型的参数
            sae_test.optimizer.apply_gradients(zip(gradients, sae_test.trainable_variables))
            # print("--------------------------train test is -------------------------------:",
            #  self.trainable_variables)
            # cost = saeTest.lossfun(batch_xs1, batch_xs2)
            # cost = sae_test.lossfun([batch_xs1, batch_xs2], is_train)
            # cost=current_loss
            cost = sae_test.lossfun([batch_xs1, batch_xs2], is_train, _init,logger)
            logger.info(f"++++++++++++++++++++++++test_train-++++++++++++++++++++++++++++++++++{cost}+++++")
            avg_cost += cost * len(sample) / n_samples
            counter += 1
            logger.info(f"++++++++++++++++++++++++test_train-++++++++++++++++++++++++++++++++++{avg_cost}++++")
        #logger.info(f"++++++++++++++++++++++++test_train-++++++++++++++++++++++++++++++++++{costs}\n")
        costs_inter.append(avg_cost)
        # print(f"-------------costs+inter:{costs_inter}-----------------------------")
        # early stopping based on the validation data/ max_steps_without_decrease of the loss value : require_improvement
        if avg_cost < best_cost:
            # save_sess = self.sess
            best_cost = avg_cost
            # print("---------------best_cost is :------------------",best_cost)
            costs += costs_inter
            last_improvement = 0
            costs_inter = []
            logger.info(f"++++++++++++++++++++++++if-best_cost-++++++++++++++++++++++++++++++++++{best_cost}+++")
            logger.info(f"++++++++++++++++++++++++if-costs-++++++++++++++++++++++++++++++++++{costs}+++")
            #[<tf.Tensor: shape=(), dtype=float32, numpy=2.7620668>]+++
        else:
            last_improvement += 1
            # costs += costs_inter
            # costs_inter = []
        # costs_inter = []
        if last_improvement > 99:
            # print("No improvement found in a while, stopping optimization.")
            # Break out from the loop.
            stop = True
            # self.sess = save_sess
        epoch += 1
        logger.info(f"++++++++++++++++++++++++last_improvement:{last_improvement}+++++")
    # costs = list([tensor_costs for tensor_costs in costs])
    # costs = [scaler_v3 for scaler_v3 in costs ]
    # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^The end of train_test^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # print("The costs in train test is ", costs)
    # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # self.sess.close()
    # tf.reset_default_graph()
    # del self.sess
    # print("---------------costs is :----------------")
    # print(costs)
    plt.figure()
    plt.plot(costs)
    logger.info(f"++++++++++++++++++++++++plot:{costs}+++++++")
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title("Learning rate =" + str(round(learning_rate, 9)))
    plt.savefig(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data12\figure\loss_curve\train_test_picture_results' + str(
            iterator) + '_test45.png')
    plt.close()
    return best_cost


def partialProcess(iterator, selected_features, inputhtseq, inputmethy, act):
    trials = {}

    fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_trials_binary_test45_' + str(
        iterator) + '.pkl'
    with open(fname, 'wb+') as fpkl:
        pass
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

    n_hidden1 = htseq_nbr
    n_hidden2 = methy_nbr

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

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    X_train1_data = htseq_sel_data
    X_train2_data = methy_sel_data
    # 对X_train1和X_train2进行拟合和转换
   # X_train1_normalized = scaler1.fit_transform(X_train1_data)
    #X_train2_normalized = scaler2.fit_transform(X_train2_data)

    ##################################################

    if htseq_nbr > 1 and methy_nbr > 1:
        # split dataset to training and test data 80%/20%
        X_train1, X_test1 = model_selection.train_test_split(X_train1_data, test_size=0.2, random_state=1)
        X_train2, X_test2 = model_selection.train_test_split(X_train2_data, test_size=0.2, random_state=1)
        sampleInput = [X_train1, X_train2]
        testInput = [X_test1, X_test2]
        # print(type(X_train2))
        # print(type(sampleInput))
        # print("===================In main :============================")
        # print(sampleInput)
        is_train = True
        n_input1 = X_train1.shape[1]
        n_input2 = X_train2.shape[1]
        print("data 1 shape of col:", n_input1)
        print("data 2 shape of col:", n_input2)
        sae = Autoencoder(n_input1, n_input2, n_hidden1, n_hidden2, iterator, activation=act)
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


        best = fmin(cross_validation_with_input, space, algo=tpe.suggest, max_evals=20, trials=trials[trial_label])
        # best = fmin(sae.cross_validation, space, algo=tpe.suggest, max_evals=3, trials=trials[trial_label],args=(sampleInput,))
        # print(best)
        with open(fname, "ab") as file:
            pickle.dump(trials[trial_label], file)

        # saeTest = Autoencoder(n_input1, n_input2, n_hidden1, n_hidden2, activation=act)
        # testMatrix = saeTest(sampleInput, is_train=True)

        loss = train_test(params=hyperopt.space_eval(space, best), testInput=testInput, iterator=iterator,logger=logger)
        f = open(
            r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_parameter_binary45_' + str(
                iterator) + '.txt',
            'a+')
        # print("---------open the file to load the best items in best-----------------------------")

        for k, v in best.items():
            f.write(str(k) + ':' + str(v) + ',')
            # print("k",str(k))
            # print("v",str(v))
        # print("The file f recording ends")
        f.write("---------------------------------------------")
        f.write(str(loss))
        f.write('\n')
        f.close()
        logger.info(f'Partial process ended {iterator}')
        del htseq_sel_data
        del methy_sel_data
        # del trials
        del sae


def parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num):
    if num_processes is None or num_processes == 0:
        num_processes = multiprocessing.cpu_count()

    process_iteration = partial(partialProcess, selected_features=selected_features, inputhtseq=inputhtseq,
                                inputmethy=inputmethy, act=act)

    with multiprocessing.Pool(processes=num_processes) as pool:

        # results = pool.map(partial(partialProcess, selected_features=selected_features, inputhtseq=inputhtseq, inputmethy=inputmethy), range(21))
        pool.map(process_iteration, range(community_num))
        #pool.map(process_iteration, [10])

if __name__ == '__main__':
    #f = open(
       # r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_parameter_binary26.txt',
       # 'w+')
    #f.close()

    #fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_trials_binary_test26.pkl'
   # with open(fname, 'wb+') as fpkl:
    #    pass
    selected_features = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\data\selected_features.csv',
        delimiter=',', skip_header=1)

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
    #community_num = 21
    num_processes = 6
    act = tf.nn.tanh
    # C:\Program Files\R\R-4.3.3\
    # 使用R的print函数来打印一个消息

    ro.r('options(stringsAsFactors = F)')
    ro.r('suppressPackageStartupMessages(library(NbClust))')
    ro.r('suppressPackageStartupMessages(library(ggplot2))')
    ro.r('suppressPackageStartupMessages(library(grid))')
    ro.r('suppressPackageStartupMessages(library(ComplexHeatmap))')
    ro.r('suppressPackageStartupMessages(library(circlize))')
    ro.r('suppressPackageStartupMessages(library(tidyverse))')
    ro.r('suppressPackageStartupMessages(library(maftools))')
    load_file = 'C:/Users/gklizh/Documents/Workspace/code_and_data12/data/spinglass/melanet_cmt.RData'

    # 将文件路径传递给R环境
    ro.r(f"load_file <- '{load_file}'")

    # 使用这个变量来读取RDS文件
    ro.r("melanet_cmt <- readRDS(load_file)")

    # 获取社区的数量
    ro.r('number_of_communities <- length(melanet_cmt)')

    # 在Python中获取值
    community_num = ro.r('number_of_communities')[0]
    print(community_num)


    parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num)

    # tanh activation function

    # trials = {}
    # run the autoencoder for communities

# for trial_label, trial in trials.items():
# print(f"\nData for {trial_label}:")
# for trial_result in trial.trials:
# print(trial_result)
