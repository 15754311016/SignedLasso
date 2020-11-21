"""
This file implements the MLP based signed network embedding
Input:
	(xa, xb, xc): A triple (xa, xb, xc) with the link between xa and xb is positive and the link between xb and xc is negative
	d: the dimesnion of the embedded space
	N: number of nodes in the graph
Output:
	X: Nd x 1, embedded representation

Obj:
	min(0, f(xb,xc) + 1 - f(xa,xc))
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T

import cPickle
import pickle
import numpy as np
import cPickle
import matplotlib.pyplot as plt

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


# start-snippet-1
class FirstHiddenLayer(object):
    """
    """
    def __init__(self, rng, input1, input2, input3, n_in, n_out, W1=None, W2=None, b=None, activation=T.tanh):
        """
        This is an variation of the typical hidden layer of a MLP. SPecifically, it accepts two inputs where each input is put into a hidden layer which share the same weights
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input1,W) + b), tanh(dot(input2,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input1: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type input2: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input1 = input1
        self.input2 = input2
        self.input3 = input3
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W1 is None:
            W1_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W1_values *= 4

            W1 = theano.shared(value=W1_values, name='W1', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W1 = W1
        self.b = b

        if W2 is None:
            W2_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W2_values *= 4

            W2 = theano.shared(value=W2_values, name='W2', borrow=True)


        self.W2 = W2

        lin_output1 = T.dot(input1, self.W1) + T.dot(input2, self.W2) + self.b
        lin_output2 = T.dot(input1, self.W1) + T.dot(input3, self.W2) + self.b
        self.output1 = (
            lin_output1 if activation is None
            else activation(lin_output1)
        )
        self.output2 = (
        	lin_output2 if activation is None
        	else activation(lin_output2)
        )
        # parameters of the model
        self.params = [self.W1, self.W2, self.b]


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input1, input2, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        This is an variation of the typical hidden layer of a MLP. SPecifically, it accepts two inputs where each input is put into a hidden layer which share the same weights
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input1,W) + b), tanh(dot(input2,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input1: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type input2: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output1 = T.dot(input1, self.W) + self.b
        lin_output2 = T.dot(input2, self.W) + self.b
        self.output1 = (
            lin_output1 if activation is None
            else activation(lin_output1)
        )
        self.output2 = (
            lin_output2 if activation is None
            else activation(lin_output2)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class SNE(object):
    """
    Signed Network Embedding
    """

    def __init__(self, rng, index1, index2, index3, n_in, n_hidden, n_out, N, d, x=None):
        """
        Initialize the parameters for signed network embedding

            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type index1: int
            :para input1: the index of xa

            :type index2: int
            :para input2: the index of xb

            :type index3: int
            :para input3: the index of xc

            :type n_in: int
            :param n_in: number of input units, the dimension of the space in
            which the datapoints lie

            :type n_hidden: int
            :param n_hidden: number of hidden units

            :type n_out: int
            :param n_out: number of output units, the dimension of the space in
            which the labels lie


            :type N: int
            :para N: number of nodes in the network

            :type d: int
            :para d: the dimesnion of each embedding

            :type x: theano.tensor.dmatrix
            :para x: the embedding of the nodes
        """


        # random initialize the signed network embedding
        if x is None:
            x_values = numpy.zeros((N,d), dtype=theano.config.floatX)
            x = theano.shared(value=x_values, name='x', borrow=True)
        self.x = x

        # The first hiddenLayer, input1 is (xa,xb), input2 is (xa,xc)
        print('1')
        self.hiddenLayer1 = FirstHiddenLayer(
            rng=rng,
            input1=self.x[index1],  # index start from zero
            input2=self.x[index2],
            input3=self.x[index3],
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        # print("_________")
        # print("index1:",index1)
        # The second hiddenLayer, input1 is self.hiddenLayer1.output1, input2 is (xa,xc), input2 is self.hiddenLayer1.output2
        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input1=self.hiddenLayer1.output1,
            input2=self.hiddenLayer1.output2,
            n_in=n_hidden,
            n_out=n_in,		# note that we only have two output values, f(xa,xb) and f(xa,xc)
            activation=T.tanh
        )

        # hidden layer 3
        self.hiddenLayer3 = HiddenLayer(
            rng=rng,
            input1=self.hiddenLayer2.output1,
            input2=self.hiddenLayer2.output2,
            n_in=n_hidden,
            n_out=1,		# note that we only have two output values, f(xa,xb) and f(xa,xc)
            activation=T.tanh
        )

        # L1 norm; one regulariza; one regularization option is to enforce L1norm to be small
        self.L1 = (
            abs(self.hiddenLayer1.W1).sum()
            + abs(self.hiddenLayer1.W2).sum()
            + abs(self.hiddenLayer2.W).sum()
            + abs(self.hiddenLayer3.W).sum()
            #+ abs(T.dot(index1,self.x))
            #+ abs(T.dot(index2,self.x))
            #+ abs(T.dot(index3,self.x))
            + self.hiddenLayer1.input1.sum()
            + self.hiddenLayer1.input2.sum()
            + self.hiddenLayer1.input3.sum()
        )

        # square of L2 norm; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer1.W1 ** 2).sum()
            + (self.hiddenLayer1.W2 ** 2).sum()
            + (self.hiddenLayer2.W ** 2).sum()
            + (self.hiddenLayer3.W ** 2).sum()
            #+ ((T.dot(index1,self.x)) ** 2).sum()
            #+ ((T.dot(index2,self.x)) ** 2).sum()
            #+ ((T.dot(index3,self.x)) ** 2).sum()
            + (self.hiddenLayer1.input1.sum() ** 2).sum()
            + (self.hiddenLayer1.input2.sum() ** 2).sum()
            + (self.hiddenLayer1.input3.sum() ** 2).sum()
        )

        # max(0, f(xa,xc)+1-f(xa,xb))

        '''
        self.objective_value = (
            -T.mean(
                T.max(T.concatenate([T.zeros(index1.shape[0],1), self.hiddenLayer2.output2 + 1 -
                                     self.hiddenLayer2.output1], axis=1), axis=1
                      )
            )
        )
        '''
        self.value = self.hiddenLayer3.output2 + 1 - self.hiddenLayer3.output1
        self.objective_value = (
            T.mean(self.value * (self.value>0))
        )

        self.params = [self.x] + self.hiddenLayer1.params + self.hiddenLayer2.params + self.hiddenLayer3.params


def shared_dataset(data_abc, borrow=True):
    """
    function that loads the dataset into shared varaible
    """
    data_a, data_b, data_c = data_abc
    shared_a = theano.shared(numpy.asarray(data_a, dtype=theano.config.floatX), borrow=borrow)
    shared_b = theano.shared(numpy.asarray(data_b, dtype=theano.config.floatX), borrow=borrow)
    shared_c = theano.shared(numpy.asarray(data_c, dtype=theano.config.floatX), borrow=borrow)

def sgd_optimization_SNE(learning_rate=0.8, L1_reg=0.00, L2_reg=0.0001, n_epochs=100, dataset='Epinions.pkl.gz',
                         batch_size=80, n_hidden=80):
    """
    Stochastic gradient descent optimization of SNE
    """
    # load dataset
    # datapath = '/home/suhang/data/lisa/data/epinions/train_data.p'
    # f = open( "/home/suhang/data/lisa/data/epinions/train_set_a_2.p", "rb" )
    # train_set_a_value = pickle.load(f)
    # f.close()
    # num_nodes = len(train_set_a_value)
    # print num_nodes
    # rand_index = np.random.permutation(num_nodes)
    #
    # f = open( "/home/suhang/data/lisa/data/epinions/train_set_b_2.p", "rb" )
    # train_set_b_value = pickle.load(f)
    # f.close()
    #
    # f = open( "/home/suhang/data/lisa/data/epinions/train_set_c_2.p", "rb" )
    # train_set_c_value = pickle.load(f)
    # f.close()
    print time.localtime(time.time())
    f = open('epinions_large.p')
    train_value = cPickle.load(f)
    print"train_value\n",train_value
    f.close()

    num_nodes = train_value.shape[0]
    print num_nodes
    rand_index = np.random.permutation(num_nodes)
    print(len(rand_index))
    print(len(np.asarray(train_value[:, 0], dtype=np.int32)[rand_index]))
    # train_set_a, train_set_b, train_set_c = shared_dataset([train_set_a_value,train_set_b_value,train_set_c_value],
    #                                                borrow=True)

    train_set_a = theano.shared(np.asarray(train_value[:,0], dtype=np.int32)[rand_index],
                                name='train_set_a',
                                borrow=True)
    train_set_b = theano.shared(np.asarray(train_value[:,1], dtype=np.int32)[rand_index], name='train_set_b',
                                borrow=True)
    train_set_c = theano.shared(np.asarray(train_value[:,2], dtype=np.int32)[rand_index], name='train_set_c',
                                borrow=True)
    # train_set_a = theano.shared(train_set_a_value, name='', borrow=True)
    # train_set_b = theano.shared(train_set_b_value, name='', borrow=True)
    # train_set_c = theano.shared(train_set_c_value, name='', borrow=True)



    # compute number of minibatches for training
    n_train_batches = train_set_a.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '...building the model'

    # allocate symbolic variables for the data
    index = T.ivector('idx')
    lr = T.scalar('lr')

    # generate symbolic variables for input
    xa = T.ivector('xa')
    xb = T.ivector('xb')
    xc = T.ivector('xc')
    rng = numpy.random.RandomState(1234)

    # construct the
    SNE_model = SNE(
        rng=rng,
        index1=xa,
        index2=xb,
        index3=xc,
        n_in=10,
        n_hidden=10,
        n_out=1,
        N=8045,
        d=10
    )
    # rng, index1, index2, index3, n_in, n_hidden, n_out, N, d, x=None

    # the cost we minimuze during training is max(0, f(xa,xc)+1-f(xa,xb))
    cost = (
        SNE_model.objective_value + L1_reg * SNE_model.L1
        + L2_reg * SNE_model.L2_sqr
    )

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in SNE_model.params]
    updates = [
        (param, param - lr * gparam)
        for param, gparam in zip(SNE_model.params, gparams)
    ]


    # compiling a Theano function 'train_model' that returns the cost, but in the same time updates the parameter of the model based on the rules defined in 'updates'
    train_model = theano.function(
        inputs=[index, lr],
        outputs=cost,
        updates=updates,
        givens={
            xa: train_set_a[index],
            xb: train_set_b[index],
            xc: train_set_c[index]
        }
    )


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # # early-stopping parameters
    # patience = 10000	# look as this many examples regardless
    # patience_increase = 2	# wait this much longer when a new best is found
    # improvement_threshold = 0.995	# a relative improvement of this much is considered significant

    epoch = 0
    done_looping = False
    error = []
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        kf = get_minibatches_idx(train_value.shape[0], batch_size, shuffle=True) # TODO: cehck this

        total_avg = 0
        for minibatch_index, train_index in kf:
        # for minibatch_index in xrange(n_train_batches):
        #     print("train_index",train_index)
        #     print("train_index",train_index.shape)

            minibatch_avg_cost = train_model(train_index, learning_rate)
            # iteration number
            total_avg += minibatch_avg_cost * len(train_index)

        total_avg = total_avg / train_value.shape[0]

        if epoch % 100 == 0:
            learning_rate = learning_rate * 0.9
	#if epoch % 100 == 0:
	print("epoch " + str(epoch) + ": " + str(total_avg))
        error.append(total_avg)
        if epoch % 500 == 0 or epoch == 1:
            filename = "train/" + str(epoch) + '.p'
            with open(filename,'wb') as fp:
                pickle.dump(SNE_model.params,fp)
            fp.close()
    #print time.localtime(time.time())
    f_11 = open('error.txt','w')
    f_11.write(str(error))
    #plt.figure()
    #plt.title('X2')
    #plt.plot(range(1000), error, 'b-', label="$lasso$")
    #plt.xlabel('epoch')
    #plt.ylabel('value')
    #plt.legend()
    #plt.show()



if __name__=='__main__':
    print("start")
    print time.localtime(time.time())
    sgd_optimization_SNE()
    print time.localtime(time.time())
#
# para = pickle.load(open('train/10.p','rb'))
# emb = np.asarray(para[0].get_value())
# W1 = np.asarray(para[1].get_value())
# W2 = np.asarray(para[2].get_value())
# b = np.asarray(para[3].get_value())
# emb1 = np.tanh(np.dot(emb,W1)+b)
# emb2 = np.tanh(np.dot(emb,W2)+b)
# print emb.shape
# print W1.shape
# print W2.shape
# print emb2.shape
