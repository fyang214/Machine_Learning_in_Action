''' P21
Pseudo code for KNN
For every point in our dataset:
calculate the distance between inX and the current point
sort the distances in increasing order
take k items with lowest distances to inX
find the majority class among these items
return the majority class as our prediction for the class of inX
'''
from numpy import *
import operator  # Operator module, which is used later in the KNN for sorting

def createDateSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

''' Parameters:
inX: data point to be classified
dataSet: data points with label marked
labels: label of dataSet. Order of labels matching dataSet
k: number of neighbours of inX to be considered
'''

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    '''
    numpy.tile(a,b) - make a matrix same as b, but each element in b is inX
    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1, 2, 1, 2],
       [3, 4, 3, 4]])
    >>> np.tile(b, (2, 1))
    array([[1, 2],
       [3, 4],
       [1, 2],
       [3, 4]])

    tile(inX, (dataSetSize,1)) create an array where data to be predicted appear as many times as training data
    inX == [2,0], and we have 3 historic data then tile(inX, (dataSetSize,1))  = [[2,0],
                                                                                 [2,0],
                                                                                 [2,0]]
    and diffMat is the difference of coordinates of historic data and inX
    '''
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()  # numpy.argsort(a, axis=-1, kind='quicksort', order=None)[source]
    # Returns the indices that would sort an array.
    '''
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])
    Two-dimensional array:

    >>>
    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
       [2, 2]])
    >>>
    >>> np.argsort(x, axis=0)  # sorts along first axis (down)
    array([[0, 1],
       [1, 0]])
    >>>
    >>> np.argsort(x, axis=1)  # sorts along last axis (across)
    array([[0, 1],
       [0, 1]])
    '''
    classCount = {}   #classCount是一个dict，key是每个label，value是k个离被预测点最近的点相应label的个数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #sortedDistIndicies 是距离离被预测点由小到大的排列序号。
                                               # 比如sortedDistIndicies第一个元素是3， 则说明第三个点是离被预测点距离最近的
        #得到最近的点的label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # python dictionary get(key, value) method returns
                                                                    # the value for the specified key if key is in dictionary.
                                                                    # None if the key is not found and value is not specified.
                                                                    # value if the key is not found and value is specified.
        #这个最近的点的label value + 1
    sortedClassCount = sorted(classCount.items(),                   # dict.items() method
                                                                    # Syntax: dictionary.items()
    #sortedClassCount is a list of tuple,                           # Parameters: This method takes no parameters.
    #each tuple got 2 items, label and number                       # Returns: A view object that displays a list of a given dictionary’s (key, value) tuple pair.
    #Example:[('B', 2), ('A', 1)]                                   # Retun example: dict_items([('C', 'Geeks'), ('B', 4), ('A', 'Geeks')])
                    key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0] #first element of the first element is the label [('B', 2), ('A', 1)]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        labels = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
        classLabelVector.append(labels[listFromLine[-1]])
        index += 1
    return returnMat, classLabelVector
