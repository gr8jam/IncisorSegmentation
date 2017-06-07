import numpy as np
from matplotlib import pyplot as plt

# x = np.arange(5)
#
# fig = 1
# f1 = plt.figure()
# h = plt.plot(x)
# plt.show(block=False)
# plt.waitforbuttonpress()
#
# f2 = plt.figure()
# h = plt.plot(x)
# plt.show(block=False)
# plt.waitforbuttonpress()
#
# plt.figure(f1.number)
# h = plt.plot(x+5)
# plt.show(block=False)
# plt.waitforbuttonpress()

A = np.array([[2.0, 5, 0], [0, 0, 1], [2, 3.0, 2.0]])
b = np.array([1, -1, 0])

# print "A shape:" + str(A.shape)
# print A
# print "b shape: " + str(b.shape)
# print b
#
# c = np.dot(A,b)
# print "c shape: " + str(c.shape)
# print c

# print A
# R = np.roll(A, 1, axis=1)
# print R
# print A


# a = np.array([0,0,1,1,2], dtype='float')
# b = np.array([0,1,0,1,3], dtype='float')
#
# with np.errstate(divide='ignore', invalid='ignore'):
#     c = np.true_divide(a,b)
#     # c[c == np.inf] = 0
#     # c = np.nan_to_num(c)
#
# print('c: {0}'.format(c))

# print c

# alpha = np.arctan2(1,0)
# print alpha

# M = np.arange(12).reshape((3,4))
#
# suma = np.sum(M, axis=0)
# print suma
#
# aa = M / suma
# print aa


# multi = np.zeros((2, 3, 4))
# # print multi
# multi[1, 0, 0] = 1
#
# print multi[:,:,:]

if True: print "Hello"