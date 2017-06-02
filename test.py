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

A = np.array([[2, 1, 0], [0, 0, 1]])
b = np.array([1, -1, 0])

# print "A shape:" + str(A.shape)
# print A
# print "b shape: " + str(b.shape)
# print b
#
# c = np.dot(A,b)
# print "c shape: " + str(c.shape)
# print c

print A
R = np.roll(A, 1, axis=1)
print R
print A
