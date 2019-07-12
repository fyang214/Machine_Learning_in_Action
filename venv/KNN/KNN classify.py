from KNN import KNN

# group, labels = KNN.createDateSet()


# print (KNN.classify0([0, 0], group, labels, 3))
from numpy import *
datingDataMat,datingLabels = KNN.file2matrix('datingTestSet.txt')
print(datingDataMat)
print(datingLabels[0:20])

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()