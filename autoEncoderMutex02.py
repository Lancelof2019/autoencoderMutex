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
    def __init__(self, n_hidden1, n_hidden2, activation):
        super(inputSmallNetwork, self).__init__()
        self.l2 = None
        self.l1 = None
        self.n_input1 = n_hidden1
        self.n_input2 = n_hidden2
        # self.is_train = is_train
        # self.concate = self.input_concate
        self.activation = activation

        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        self.l1_layer = tf.keras.layers.Dense(self.n_input1, kernel_initializer='random_normal', name='layer1')
        self.l2_layer = tf.keras.layers.Dense(self.n_input2, kernel_initializer='random_normal', name='layer2')

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

    def get_weights(self):
        if not self.l1_layer.built or not self.l2_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l1_layer.kernel, self.l2_layer.kernel,self.l1_layer.bias,self.l2_layer.bias



class encoderNetwork(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, n_hiddensh, activation):
        super(encoderNetwork, self).__init__()
        # super().__init__(*args, **kwargs)
        self.l3 = None
        self.ensmallNetwork = inputSmallNetwork(n_input1, n_input2, activation)
        self.n_hidden3 = n_hiddensh
        self.l3_layer = tf.keras.layers.Dense(self.n_hidden3, kernel_initializer='random_normal', name='layer3')

        # l3 = self.l3_layer(tf.concat([self.small_network.l1,self.small_network.l2 ], 1))
        # l1,l2 are bounded in l
        # print("-----layer3 shape", l3.shape)
        # l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))

        # self.concatenate = tf.keras.layers.Concatenate()
        # self.output_layer = tf.keras.layers.Dense(self.concate, activation='softmax')

    def call(self, inputs, is_train):
        self.is_train = is_train
        output = self.ensmallNetwork(inputs, self.is_train)

        # self.l3_layer = tf.keras.layers.Dense(self.n_hiddensh, kernel_initializer=self.init, name='layer3')
        l3 = self.l3_layer(tf.concat([output[0], output[1]], 1))
        l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))

        self.l3 = l3
        # self.Wsht = self.l3_layer.kernel
        return self.l3

    def get_weights(self):
        if not self.l3_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l3_layer.kernel,self.l3_layer.bias


class outputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, n_input1, n_input2, activation):
        super(outputSmallNetwork, self).__init__()
        self.l5 = None
        self.l6 = None
        self.n_hidden5 = n_input1
        self.n_hidden6 = n_input2
        self.activation = activation
        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        self.l5_layer = tf.keras.layers.Dense(self.n_hidden5, kernel_initializer='random_normal', name='layer1')
        self.l6_layer = tf.keras.layers.Dense(self.n_hidden6, kernel_initializer='random_normal', name='layer2')

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

    def get_weights(self):
        if not self.l6_layer.built or not self.l5_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l5_layer.kernel, self.l6_layer.kernel,self.l5_layer.bias, self.l6_layer.bias


class decoderNetwork(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, n_hidden1, n_hidden2, activation):
        super(decoderNetwork, self).__init__()

        self.l4_layer = tf.keras.layers.Dense(n_hidden1 + n_hidden2, kernel_initializer='random_normal',
                                              name='layer4')

        # self.l5_layis_trainer = tf.keras.layers.Dense(input_n_hidden1, kernel_initializer=self.init, name='layer5')
        # self.l6_layer = tf.keras.layers.Dense(input_n_hidden2, kernel_initializer=self.init, name='layer6')
        self.n_hidden5 = n_input1
        self.n_hidden6 = n_input2
        self.outsmallNetwork = outputSmallNetwork(self.n_hidden5, self.n_hidden6, activation)

    def call(self, inputs, is_train):
        l4 = self.l4_layer(inputs)
        self.is_train = is_train
        # self.is_train = True
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))

        self.l4 = l4
        # self.Wsh = self.l4_layer.kernel
        output = tf.split(l4, [self.n_hidden5, self.n_hidden6], 1)
        l5, l6 = self.outsmallNetwork(output, self.is_train)
        return l5, l6

    def get_weights(self):
        if not self.l4_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l4_layer.kernel,self.l4_layer.bias



# -------------------------------------------------------------------------

class Autoencoder(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, n_hidden1, n_hidden2, activation):
        super(Autoencoder, self).__init__()
        self.n_hiddensh =1
        self.encoder = encoderNetwork(n_hidden1, n_hidden2, self.n_hiddensh, activation)
        self.decoder = decoderNetwork(n_input1, n_input2, n_hidden1, n_hidden2, activation)
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.max_epochs =1000
        self.require_improvement = 30
        self.lamda = 0.13
        self.alpha = 0.012
        self.learning_rate = 0.032
        # self.inputData=X_train

    def call(self, inputs, is_train):
        self.sampleInput = inputs
        self.temp_record = inputs
        #print("The first time of sampleInput",type(sampleInput))
        self.is_train = is_train
        encoded = self.encoder(inputs, self.is_train)
        decoded = self.decoder(encoded, self.is_train)

        self.W1, self.W2,self.bias1,self.bias2= self.encoder.ensmallNetwork.get_weights()
        # self.W2 = self.encoder.ensmallNetwork.
        self.Wsht,self.bias3= self.encoder.get_weights()
        self.Wsh,self.bias4= self.decoder.get_weights()
        self.W1t, self.W2t ,self.bias5,self.bias6= self.decoder.outsmallNetwork.get_weights()

        return  decoded
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
        #return decoded

    def L1regularization(self, weights):
        return tf.reduce_sum(tf.abs(weights))

    def L2regularization(self, weights, nbunits):
        return math.sqrt(nbunits) * tf.nn.l2_loss(weights)
    def lossfun(self,sampleInput,is_train):

        # self.H = self.encodefun(X1, X2)
        # X1_, X2_ = self.decodefun(self.H)
        # self.get_weights()
        #print("The 2nd time of sampleInput", type(sampleInput))
        self.compareOutPut = self.call(sampleInput,is_train)
        #print("The 3rd time of sampleInput", type(sampleInput))
        sgroup_lasso = self.L2regularization(self.W1, self.n_input1 * self.n_hidden1) + \
                       self.L2regularization(self.W2, self.n_input2 * self.n_hidden2)
        # print(sgroup_lasso.shape)
        # lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)

        error = tf.reduce_mean(tf.square(sampleInput[0] - self.compareOutPut[0])) + tf.reduce_mean(
            tf.square(sampleInput[1] - self.compareOutPut[1]))
        cost = 0.5 * error + 0.5 * self.lamda * (1 - self.alpha) * sgroup_lasso + 0.5 * self.lamda * self.alpha * lasso
        return cost
    #def optiGradient(self):
    #def train_step(self,batch_xs1, batch_xs2):
         #is_train=True
         #with tf.GradientTape() as tape:
         #   current_loss = self.lossfun(batch_xs1, batch_xs2)
         #gradients = tape.gradient(current_loss, self.trainable_variables)
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
         #return current_loss

    def train(self,inputs):


        # save_sess = self.sess

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

        #n_samples = train_input1.shape[0]  # size of the training set #370 x 4/5
        #vn_samples = val_input1.shape[0]  # size of the validation set#370 x 1/5
        #print(type(self.sampleInput))
        #print(self.sampleInput[0].numpy())
        n_samples = self.sampleInput[0].shape[0]
        epoch = 0
        counter = 0
        self.batch_size=16
        #k = 5
        while epoch < self.max_epochs and stop == False:
            # for(self.max)
            # train the model on the training set by mini batches
            # shuffle then split the training set to mini-batches of size self.batch_size
            #logging.info(
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
            #print(self.sampleInput[0].shape)
            #print(self.sampleInput[1].shape)

            logging.info(
                "----------------------one trial for train samples starts one epoch ------------------------\n")
            for sample in mini_batches:
                #print("#############Sample:", len(sample))
                #s1 = self.sampleInput[0].numpy()
                #s2 = self.sampleInput[1].numpy()
                #print(s1)
                #print(s2)
                #print(type(inputs))
                #print(type(inputs[0]))
                batch_xs1 = inputs[0][sample][:]
                batch_xs2 = inputs[1][sample][:]
                #batch_xs1 = s1[sample][:]
                #batch_xs2 = s2[sample][:]

                #batch_xs1 = self.sampleInput[0][sample][:]
                #batch_xs2 = self.sampleInput[1].numpy()[sample][:]


                self.is_train = True
                #loss = self.train_step(batch_xs1, batch_xs2)
                # feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: True}
                # cost = self.sess.run([self.loss_, self.train_step], feed_dict=feed_dictio)
                # avg_cost += cost[0] * len(sample) / n_samples

                with tf.GradientTape() as tape:
                    # 计算当前批次的损失
                    #if epoch == 0 and counter == 0:
                        #current_loss = self.loss(batch_xs1, batch_xs2)
                    #else:
                     current_loss = self.lossfun([batch_xs1, batch_xs2],self.is_train)
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
                cost = self.lossfun([batch_xs1, batch_xs2],self.is_train)
                # print("---------------------train costs-------------")
                # print("cost:",cost)

                # print("-----------------------Kosten---------------------------",cost)
                # print("---------------------------------------------")
                avg_cost += cost * len(sample) / n_samples
                counter += 1
                # print("----------------------------cost_train:", cost, avg_cost, "-----------------------------")
            # print("---------------------train costs ends-------------")
            costs_inter.append(avg_cost)
            if avg_cost < best_cost_cost:
                best_cost_cost = avg_cost
                costs += costs_inter
                last_improvement = 0
                costs_inter = []
            else:
                last_improvement += 1

            if last_improvement > self.require_improvement:
                stop = True
            epoch += 1
        final_loss = costs[-1]
        plt.figure()
        plt.plot(costs)
        plt.ylabel('cost Loss')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(round(self.learning_rate, 9)))
        ########################################################
        plt.annotate(f'Final Loss: {final_loss:.9f}',
                     xy=(len(costs) - 1, final_loss),
                     xytext=(len(costs) / 2, final_loss),
                     textcoords='data',
                     #arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='top')

        #plt.text(len(costs) - 1, final_loss, f'Final Loss: {final_loss:.2f}', fontsize=10, ha='right', va='bottom')

        #########################################################
        plt.savefig(r'C:\Users\gklizh\Documents\Workspace\code_and_data12\figure\loss_curve\training_picture' + f'_testW.png')
        plt.close()

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

    # start_time = time.perf_counter()
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

    iterator = 0
    # ------------------------------------------------------------------------------------------------------------
    # print('iteration', iterator)
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
        sampleInput = [X_train1, X_train2]
        is_train = True
        n_input1 = X_train1.shape[1]
        n_input2 = X_train2.shape[1]
        print("data 1 shape of col:", n_input1)
        print("data 2 shape of col:", n_input2)
        sae = Autoencoder(n_input1, n_input2, n_hidden1, n_hidden2, activation=act)
        resultMatrix = sae(sampleInput, is_train)
        sae.train(sampleInput)

        # print(resultMatrix)
        #print(resultMatrix[0].numpy().shape)
        #print(resultMatrix[1].numpy().shape)
