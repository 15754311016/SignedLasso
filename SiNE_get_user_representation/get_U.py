import numpy as np
import pickle

import scipy
#
# import theano
# # import theano.tensor as T
# #
# # import cPickle
# # import pickle
# a = np.random.randn(10,2)
# # print(a)
# print(np.array([np.linalg.norm(w) for w in a]))
# print(np.linalg.norm(a,axis=1,keepdims=True).reshape(10,))
#
# score = a.max(1)
# print(score)
# idx = np.argsort(score, 0)
# idx = idx[::-1]
# print(idx)
para = pickle.load(open('train/2000.p', 'rb'))
print(type(para[0]))
emb = np.asarray(para[0].get_value())
print(emb[1000])
np.savetxt('U_sine.txt',emb[1:],encoding='utf-8',fmt='%.10f')
# scipy.io.savemat(filename, {'emb': emb, 'emb1': emb1, 'emb2': emb2})
