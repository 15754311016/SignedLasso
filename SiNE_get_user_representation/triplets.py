import numpy
import scipy.io
import cPickle

A_p = numpy.matrix(numpy.loadtxt('B_p.csv', delimiter=','))
A_n = numpy.matrix(numpy.loadtxt('B_n.csv', delimiter=','))
A = A_p - A_n
# print(type(A))
# print(A.shape)
# B =  numpy.matrix(numpy.ones((1226,613)))
# C =  numpy.matrix(numpy.ones((1226,613)))
# A = numpy.hstack((B,C))
# print(type(A))
# testA = mat['testA'].todense()

threshold = 200

triplets = []

for i in xrange(A.shape[0]):
    poss = numpy.where(A[i, :] == 1)[1]
    # print"poss:",poss
    negs = numpy.where(A[i, :] == -1)[1]
    # print'negs:',negs
    tuples = []
    for pos_ind in poss:
        for neg_ind in negs:
            tuples.append((pos_ind + 1, neg_ind + 1))

    if len(negs) == 0:
        for pos_ind in poss:
            tuples.append((pos_ind + 1, 0))

    if len(tuples) > threshold:
        randind = numpy.random.permutation(len(tuples))[0:threshold]
        newtuples = []
        for ind in randind:
            newtuples.append(tuples[ind])
        tuples = newtuples

    for tuple in tuples:
        triplets.append([i + 1, tuple[0], tuple[1]])

triplets = numpy.asarray(triplets, dtype='int32')
print triplets.shape
a = list(triplets[:, 0])
b = list(triplets[:, 1])
a.extend(b)
print(len(set(a)))

cPickle.dump(triplets, open('epinions_large.p', 'w'))
