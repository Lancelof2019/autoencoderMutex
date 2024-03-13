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
import tensorflow as tf
import pickle
import math
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# 定义一个小的神经网络层
class inputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, n_hidden1, n_hidden2, activation,_init):
        super(inputSmallNetwork, self).__init__()
        self.l2 = None
        self.l1 = None
        self.n_input1 = n_hidden1
        self.n_input2 = n_hidden2
        # self.is_train = is_train
        # self.concate = self.input_concate
        self.activation = activation

        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
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
        #print("The   self.n_input1 = n_hidden1 is :", self.n_input1)
        #print("The   self.n_input2 = n_hidden2 is :", self.n_input2)
        self.l1 = l1
        self.l2 = l2
        # self.W1 = self.l1_layer.kernel
        # self.W2 = self.l2_layer.kernel
        return self.l1, self.l2

    def get_weights(self):
        if not self.l1_layer.built or not self.l2_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l1_layer.kernel, self.l2_layer.kernel, self.l1_layer.bias, self.l2_layer.bias


# 定义一个大的神经网络层，包含三个小的神经网络层
class encoderNetwork(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, n_hiddensh, activation,_init):
        super(encoderNetwork, self).__init__()
        # super().__init__(*args, **kwargs)
        self.l3 = None
        #self._init=_init
        self.ensmallNetwork = inputSmallNetwork(n_input1, n_input2, activation,_init)
        #print("The weights in layers 1 in encoderNetwork",self.ensmallNetwork.l1_layer.get_weights())

        self.n_hidden3 = n_hiddensh
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
        self.W1=self.ensmallNetwork.l1_layer.kernel
        self.W2=self.ensmallNetwork.l2_layer.kernel
        self.bias1=self.ensmallNetwork.l1_layer.bias
        self.bias2= self.ensmallNetwork.l2_layer.bias

        #print("The weights in layers 1 in encoderNetwork", self.ensmallNetwork.l1_layer.get_weights())
        #print("The wieights in layer1 is :",self.ensmallNetwork.l1_layer.kernel)
        # self.l3_layer = tf.keras.layers.Dense(self.n_hiddensh, kernel_initializer=self.init, name='layer3')
        l3 = self.l3_layer(tf.concat([output[0], output[1]], 1))
        l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))

        self.l3 = l3
        self.Wsht = self.l3_layer.kernel
        self.bias3 = self.l3_layer.bias
        #print("self.l3 shape is :",l3.shape)
        # self.Wsht = self.l3_layer.kernel
        return self.l3

    def get_weights(self):
        if not self.l3_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l3_layer.kernel, self.l3_layer.bias

#####################################################################################################
class outputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, n_input1, n_input2, activation,_init):
        super(outputSmallNetwork, self).__init__()
        self.l5 = None
        self.l6 = None
        self.n_hidden5 = n_input1
        self.n_hidden6 = n_input2
        self.activation = activation
        #print("The self.n_hidden5 is :",self.n_hidden5)
        #print("The self.n_hidden6 is :", self.n_hidden6)
        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        self.l5_layer = tf.keras.layers.Dense(self.n_hidden5, kernel_initializer = _init, name='layer5')
        self.l6_layer = tf.keras.layers.Dense(self.n_hidden6, kernel_initializer = _init, name='layer6')

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

       # self.W1t, self.W2t, self.bias5, self.bias6 = self.decoder.outsmallNetwork.get_weights()



        self.l5 = l5
        self.l6 = l6



        # self.W1t = self.l5_layer.kernel
        # self.W2t = self.l6_layer.kernelcosts
        return self.l5, self.l6

    def get_weights(self):
        if not self.l6_layer.built or not self.l5_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l5_layer.kernel, self.l6_layer.kernel, self.l5_layer.bias, self.l6_layer.bias


class decoderNetwork(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, n_hidden1, n_hidden2, activation,_init):
        super(decoderNetwork, self).__init__()

        self.l4_layer = tf.keras.layers.Dense(n_hidden1 + n_hidden2, kernel_initializer=_init,
                                              name='layer4')
        #print("decoderNetwork split n_hidden1:",n_hidden1)
        #print("decoderNetwork split n_hidden2:", n_hidden2)
        #print("decoderNetwork split n_input1:", n_input1)
        #print("decoderNetwork split n_input2:", n_input2)
        # self.l5_layis_trainer = tf.keras.layers.Dense(input_n_hidden1, kernel_initializer=self.init, name='layer5')
        # self.l6_layer = tf.keras.layers.Dense(input_n_hidden2, kernel_initializer=self.init, name='layer6')
        self.n_hidden5 = n_input1
        self.n_hidden6 = n_input2
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.outsmallNetwork = outputSmallNetwork(self.n_hidden5, self.n_hidden6, activation,_init)

    def call(self, inputs, is_train):
        l4 = self.l4_layer(inputs)
        self.is_train = is_train
        # self.is_train = True
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))

        self.l4 = l4
        # self.Wsh = self.l4_layer.kernel
        output = tf.split(l4, [self.n_hidden1, self.n_hidden2], 1)
        self.Wsh =self.l4_layer.kernel
        self.bias4 = self.l4_layer.bias
        l5, l6 = self.outsmallNetwork(output, self.is_train)

        #= self.l5_layer.kernel

        #self.bias5 = self.l5_layer.bias

        #self.W2t = self.l6_layer.kernel

        #self.bias6 = self.l6_layer.bias
        self.W1t = self.outsmallNetwork.l5_layer.kernel
        self.bias5 = self.outsmallNetwork.l5_layer.bias
        self.W2t = self.outsmallNetwork.l6_layer.kernel
        self.bias6 = self.outsmallNetwork.l6_layer.bias


        return l5, l6

    def get_weights(self):
        if not self.l4_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l4_layer.kernel, self.l4_layer.bias


# 定义一个更大的神经网络层，包含四个LargeNetwork实例
# -------------------------------------------------------------------------

class Autoencoder(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, activation):
        super(Autoencoder, self).__init__()
        self.is_train = None
        self.n_hiddensh = 1
        #self.encoder = encoderNetwork(n_hidden1, n_hidden2, self.n_hiddensh, activation)
        #self.decoder = decoderNetwork(n_input1, n_input2, n_hidden1, n_hidden2, activation)
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_hidden1 = None
        self.n_hidden2 = None
        # self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.max_epochs = 1000
        self.require_improvement = 500
        #self.iterator = iterator
        self.activation = activation
        # self.lamda = 0.13
        # self.alpha = 0.012
        # self.learning_rate = 0.032
        # self.inputData=X_train
        self.batch_size = None
        self.n_hidden1 = None
        self.n_hidden2 = None
        self.alpha = None
        self.lamda = None
        self.learning_rate = None
        self._init= None
        self.trigger = False

    def call(self, inputs, is_train):


        #self.sampleInput = inputs
        # print("#################################################################")
        # print(inputs)
        # print("__________________________________________________")
        # print(self.sampleInput)
        # print("#################################################################")
        # self.temp_record = inputs
        # print("The first time of sampleInput",type(sampleInput))
        self.is_train = is_train
        encoded = self.encoder(inputs, self.is_train)
        #print("L3 is :",encoded)

        decoded = self.decoder(encoded, self.is_train)
        #print("L5 and L6 are :",decoded)


        self.encoded = encoded

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

    def lossfun(self, sampleInput, is_train):

        # self.H = self.encodefun(X1, X2)
        # X1_, X2_ = self.decodefun(self.H)
        # self.get_weights()
        # print("The 2nd time of sampleInput", type(sampleInput))
        ############################################
        #print("sample input is:",sampleInput)
        if self.trigger ==False:
            self.encoder = encoderNetwork(self.n_hidden1, self.n_hidden2, self.n_hiddensh, self.activation,self._init)
            self.decoder = decoderNetwork(self.n_input1, self.n_input2, self.n_hidden1, self.n_hidden2, self.activation, self._init)
            self.trigger = True
        #print("self.trigger is :",self.trigger)
        #############################################
        self.compareOutPut = self.call(sampleInput, is_train)

        #print("self.compareOutPut is :", self.compareOutPut)
        # print("The 3rd time of sampleInput", type(sampleInput))
        sgroup_lasso = self.L2regularization(self.W1, self.n_input1 * self.n_hidden1) + \
                       self.L2regularization(self.W2, self.n_input2 * self.n_hidden2)
        #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #print("The big value is :",self.L2regularization(self.W1, self.n_input1 * self.n_hidden1))
        #print("self.W1 is:",self.W1)
        #print(self.L2regularization(self.W1, self.n_input1 * self.n_hidden1))
        #print("self.2 is:", self.W2)
        #print(self.L2regularization(self.W2, self.n_input2 * self.n_hidden2))
        # print(sgroup_lasso.shape)
        # lasso
        #print("sgroup_lasso is ",sgroup_lasso)
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)

        #print("lasso is :",lasso)
        error = tf.reduce_mean(tf.square(sampleInput[0] - self.compareOutPut[0])) + tf.reduce_mean(
            tf.square(sampleInput[1] - self.compareOutPut[1]))
        #print("Error is:",error)
        cost = 0.5 * error + 0.5 * self.lamda * (1 - self.alpha) * sgroup_lasso + 0.5 * self.lamda * self.alpha * lasso
        #print("The cost in loss function is:", cost)
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
    def train(self, inputs,iterator):
        # training data

        # inputs = [self.inputdata[0].numpy(),self.inputdata[1].numpy()]

        train_input1 = inputs[0]  # 370 x 4/5
        train_input2 = inputs[1]

        #logging.info("---------------TRain starts---------")
        # validation data
        #val_input1 = inputs[0][val_index, :]  # 370 x 1/5
        #val_input2 = inputs[1][val_index, :]

        # costs history:
        costs = []

        costs_inter = []

        # for early stopping:
        best_cost = 10000

        stop = False
        last_improvement = 0
        # vn_samples = val_input1.shape[0]
        # n_samples = self.sampleInput[0].shape[0]
        n_samples = train_input1.shape[0]  # size of the training set #370 x 4/5
        #print("The len of n_samples:",n_samples)
        #vn_samples = val_input1.shape[0]  # size of the validation set#370 x 1/5
        epoch = 0
        counter = 0
        # self.batch_size=16
        # k = 5
        while epoch < self.max_epochs and stop == False:
            # for(self.max)
            # train the model on the training set by mini batches
            # shuffle then split the training set to mini-batches of size self.batch_size
            # logging.info(
            # f"#################################################epoch :{epoch}#################################################")
            seq = list(range(n_samples))  # 370 x 4/5
            # print("The number of n_samples",n_samples)#370 x 4/5
            random.shuffle(seq)
            mini_batches = [
                seq[k:k + self.batch_size]
                for k in range(0, n_samples, self.batch_size)
            ]

            avg_cost = 0.  # the average cost of mini_batches
            # print(self.sampleInput[0].shape)
            # print(self.sampleInput[1].shape)

            logging.info(
                "----------------------one trial for train samples starts one epoch ------------------------\n")
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
                    current_loss = self.lossfun([batch_xs1, batch_xs2], self.is_train)
                logging.info(
                    f"----------------------check if the weighths  added into tape loss in this patch -----{epoch}-------------------------")
                logging.info(self.trainable_variables)
                logging.info(
                    "-----------------------------------------patch ends in tape loss -----------------------------------------")
                # 计算梯度
                gradients = tape.gradient(current_loss, self.trainable_variables)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # print(gradients)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # 更新模型的参数
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                logging.info(
                    f"----------------------check if the weighths has been approved in this patch -----{epoch}-------------------------")
                # logging.info(self.trainable_variables)
                logging.info(
                    "-----------------------------------------patch ends in train-----------------------------------------")
                cost = self.lossfun([batch_xs1, batch_xs2], self.is_train)
                #print("The cost of one sample",cost)
                # print("---------------------train costs-------------")
                # print("cost:",cost)

                # print("-----------------------Kosten---------------------------",cost)
                # print("---------------------------------------------")
                avg_cost += cost * len(sample) / n_samples
                counter += 1
                #print("The avg_cost of one sample with accumulation", avg_cost)
                # print("----------------------------cost_train:", cost, avg_cost, "-----------------------------")
            # print("---------------------train costs ends-------------")
            ##########################################################################################################

            # logging.info("\n")
            costs_inter.append(avg_cost)

            #####################################################################################################
            if avg_cost < best_cost:

                best_cost = avg_cost
                costs += costs_inter  # costs history of the training set

                # print("show me the costs",costs)

                # print("show me the costs_val",costs_val)
                # print("###########################################")
                last_improvement = 0

                costs_inter = []

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

        costfinal = self.lossfun([train_input1 , train_input2], is_train=False)
        print("costs:\n",costs)
        print("costfinal:\n",costfinal)
        res = self.encoded
        plt.figure()
        plt.plot(costs)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(round(self.learning_rate, 9)))
        plt.savefig(r'C:\Users\gklizh\Documents\Workspace\code_and_data12\figure\loss_curve\getscores_picture_' + str(iterator) + '_test45.png')
        plt.close()
        return costfinal, res
    #################################################################################################


def Main(inputs,iterator,params,obj):
    # retrieve parameters
    global _init, optimizer
    batch_size = params['batch_size']
    # print("self.batch_size",self.batch_size)#16
    n_hidden1 = params['units1']
    n_hidden2 = params['units2']
    alpha = params['alpha']
    lamda = params['lamda']
    learning_rate = params['learning_rate']

    obj.batch_size = batch_size
    obj.n_hidden1 =  n_hidden1
    obj.n_hidden2 = n_hidden2
    obj.alpha = alpha
    obj.lamda = lamda
    obj.learning_rate = learning_rate
    # k fold validation
    k = 5
    require_improvement = 1000
    max_epochs = 1000
    init = params['initializer']
    if init == 'normal':
        _init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
    if init == 'uniform':
        _init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
    current_time = int(time.time())
    if init == 'He':
        _init = tf.keras.initializers.HeNormal(seed=current_time)
    if init == 'xavier':
        _init = tf.keras.initializers.GlorotNormal(seed=current_time)


    opt = params['optimizer']
    print("The self.n_hidden1 is :", n_hidden1)
    print("The self.n_hidden2 is :", n_hidden2)
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
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9)
    if opt == 'RMSProp':
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)

    obj.optimizer = optimizer
    obj._init = _init

    k = 5
    print("H.layer1:", n_hidden1, ", H.layer2:", n_hidden2)
    print("k", k, "lamda", lamda, ", batch_size:", batch_size, ", alpha:", alpha, ", learning_rate:",
          learning_rate)
    print("initializer: ", init, ', optimizer:', opt)

    # tf.reset_default_graph()

    # self.X1 = tf.placeholder("float", shape=[None, self.training_data1.shape[1]])
    # self.X2 = tf.placeholder("float", shape=[None, self.training_data2.shape[1]])
    #self.is_train = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool, name="is_train")

    # self.loss_ = self.loss(self.X1, self.X2)

    # if opt == 'Momentum':
    # self.train_step = self.optimizer(self.learning_rate, 0.9).minimize(self.loss_)
    # else:
    # self.train_step = self.optimizer(self.learning_rate).minimize(self.loss_)
    # Initiate a tensor session
    # init = tf.global_variables_initializer()
    # self.sess = tf.Session()
    # self.sess.run(init)

    # train the model
    #self.call(inputs,True)
    loss, res = obj.train(inputs, iterator)


    # e = shap.DeepExplainer(([self.X1, self.X2], self.H),
    # [self.training_data1, self.training_data2],
    # session=self.sess, learning_phase_flags=[self.is_train])

    # input_data1 = tf.keras.Input(shape=(self.training_data1.shape[1],))
    # input_data2 = tf.keras.Input(shape=(self.training_data2.shape[1],))
    # model = self.modelCreate(input_data1,input_data2)
    # e = shap.DeepExplainer(model, data=[self.training_data1, self.training_data2])
    # shap_values = e.shap_values([self.training_data1, self.training_data2])
    # self.sess.close()
    # tf.reset_default_graph()
    # del self.sess
    # return tloss, tres, shap_values
    return loss, res


def partialProcess(iterator, selected_features, inputhtseq, inputmethy, act):
    trials = {}
    cmt_scores = None
    fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_trials_binary_test45_' + str(
        iterator) + '.pkl'
    #with open(fname, 'wb+') as fpkl:
        #pass
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

    #n_hidden1 = htseq_nbr
    #n_hidden2 = methy_nbr

    X_train1 = htseq_sel_data
    X_train2 = methy_sel_data

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    # 对X_train1和X_train2进行拟合和转换
   # X_train1_normalized = scaler1.fit_transform(X_train1)
    #X_train2_normalized = scaler2.fit_transform(X_train2)
    sampleInput = [X_train1, X_train2]

    input = open(fname, 'rb')
    if htseq_nbr > 1 and methy_nbr > 1:
        # split dataset to training and test data 80%/20%
        trials = pickle.load(input)



        is_train = True
        n_input1 = X_train1.shape[1]#301
        n_input2 = X_train2.shape[1]

        sae = Autoencoder(n_input1, n_input2, activation=act)
        #trainMatrix = sae(sampleInput, is_train)
        best_loss = 1000
        best = trials.best_trial['result']['params']
        loss, h = Main(inputs = sampleInput, iterator = iterator, params = best, obj = sae)
        loss = loss.numpy()
        h = h.numpy()
        cmt_scores=h
        #if loss < best_loss:
           # best_loss = loss
            #best_h = h
            # best_shapfeat = shapfeat

           # if iterator == 0:
               # cmt_scores = best_h
           # else:
               # cmt_scores = np.concatenate((cmt_scores, best_h), axis=1)
        del htseq_sel_data
        del methy_sel_data
        del sae

        # fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\shap_doc\cmt_SHAPValues.pkl'
        # pickle.dump(shapfeat, open(fname, "wb"))
        with open(
            r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\community\communityScores_compare45_'+str(iterator)+'.csv',
            'w',newline='', encoding='utf-8') as csvfile:
             writer = csv.writer(csvfile, lineterminator='\n')
             [writer.writerow(r) for r in cmt_scores]



def parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act,community_num):
    if num_processes is None or num_processes == 0:
        num_processes = multiprocessing.cpu_count()

    process_iteration = partial(partialProcess, selected_features=selected_features, inputhtseq=inputhtseq,
                                inputmethy=inputmethy, act=act)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用pool.map或pool.apply_async来并行执行函数
        # 注意：由于multiprocessing的限制，不能直接传递非顶级定义的函数
        # 因此，确保process_iteration是在模块级别定义的
        # results = pool.map(partial(partialProcess, selected_features=selected_features, inputhtseq=inputhtseq, inputmethy=inputmethy), range(21))
        pool.map(process_iteration, range(community_num))#range(community_num)


if __name__ == '__main__':
    #f = open(
        #r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_parameter_binary26.txt',
        #'w+')
    #f.close()

    #fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_trials_binary_test26.pkl'
    #with open(fname, 'wb+') as fpkl:
       # pass
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

    num_processes = 3
    act = tf.nn.tanh
    community_num=21
    parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act,community_num)

    # tanh activation function

    # trials = {}
    # run the autoencoder for communities

# for trial_label, trial in trials.items():
# print(f"\nData for {trial_label}:")
# for trial_result in trial.trials:
# print(trial_result)
