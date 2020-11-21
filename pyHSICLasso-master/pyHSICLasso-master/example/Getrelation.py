
import pandas as pd
import numpy as np

# X = np.array([[1,2],[3,4],[5,6]])
# print(new_kernel(X))
#
# print(kernel_gaussian(X.T,X.T,1))
#
# A = np.arange(1,10)
# print(A)
# print(A.shape)
# print(A[None,:].shape)
import linecache
f = open('wiki3.txt','r',encoding='utf-8')
A = f.readlines()
print('data has been read')
# B = []
user_list = []
vote_list = []
user_vote_list = []
for i in range(0,len(A),8):
    print('reading line'+ str(i))
    if A[i] != 'SRC:\n':
        if A[i][4:-1] not in user_list:
            user_list.append(A[i][4:-1])
        if A[i+1][4:-1] not in user_list:
            user_list.append(A[i+1][4:-1])
            # vote_list.append(int(A[i+2][4:-1]))
        user_vote_list.append([user_list.index(A[i][4:-1]),user_list.index(A[i+1][4:-1]),int(A[i+2][4:-1])])

# user_list_sort.sort(key=user_list.index)
n = len(user_list)
# print(n)
print(len(user_list))
f = open('user_list.txt','w',encoding='utf-8')
f.write(str(user_list))
f.close()
print("start computing B:")
B_p= np.zeros((n,n),dtype='int')
B_n= np.zeros((n,n),dtype='int')
for link in user_vote_list:
    print("正在计算B")
    if link[2] == 1:
        B_p[link[0],link[1]] = 1
        B_p[link[1],link[0]] = 1
    if link[2] == -1:
        B_n[link[0],link[1]] = 1
        B_n[link[1],link[0]] = 1
np.savetxt("B_p.csv", B_p,fmt='%d',delimiter = ',')
np.savetxt("B_n.csv", B_n,fmt='%d',delimiter = ',')
# for link in user_vote_list:
#
#

# print(B.shape)

# for i in range(n):

# for i in range(0,len(A),8):
#     if A[i][4:-1]


# print(count)C = len(set(B))
# print(C)
#
# C.sort(key = B.index)
# print(C)
# print(len(C))

