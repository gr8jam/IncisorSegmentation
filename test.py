
import numpy as np
from matplotlib import pyplot as plt



x = np.arange(5)

fig = 1
f1 = plt.figure()
h = plt.plot(x)
plt.show(block=False)
plt.waitforbuttonpress()

f2 = plt.figure()
h = plt.plot(x)
plt.show(block=False)
plt.waitforbuttonpress()

plt.figure(f1.number)
h = plt.plot(x+5)
plt.show(block=False)
plt.waitforbuttonpress()
