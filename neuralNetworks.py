from pybrain.structure import FeedForwardNetwork, FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet

from random import randint, shuffle
from pylab import imshow
from scipy import io
from numpy import unique, c_, ones, zeros, hstack, argmax

import matplotlib.pyplot as plt


def plotData(X, Y, c):
    '''plots the input data '''

    # plot any one case (20x20 image) from the input
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    inputImg = X[c,:]
    imshow((inputImg.reshape(20,20)).T, cmap = 'Greys')
    
    #plot the same ouptut case
    print('the digit printed is', Y[c][0])


def convertToOneOfMany(Y):
    '''converts Y to One of many type '''
    
    rows, cols = Y.shape
    classes = unique(Y).size
    Y1 = zeros((rows, classes))

    for i in range(0, rows):
        Y1[i, Y[i]] = 1

    return Y1

# ============= load the data================
    
data = io.loadmat('ex4data1.mat')
    
# making X and Y numpy arrays
X = data['X']
Y = data['y']

# changing identity of digit '0' from 10 to 0 in array
Y[Y==10] = 0

# getting the no. of different classes in the output    
numOfLabels = unique(Y).size    
    
# ============= plotting ================

print('plotting a random digit from the input')
randomIndex = randint(0, X.shape[0])
plotData(X, Y, randomIndex)
plt.show()
    
# ============= building the dataset ================

X = c_[ones(X.shape[0]), X]
numOfExamples, sizeOfExample = X.shape

# converting Y to One Of Many type
Y = convertToOneOfMany(Y)

# separating training and test dataset
X1 = hstack((X, Y))
shuffle(X1)

X = X1[:, 0:sizeOfExample]
Y = X1[:, sizeOfExample : X1.shape[1]]

# making the classfication datasets
trainData = ClassificationDataSet(sizeOfExample, numOfLabels)
testData = ClassificationDataSet(sizeOfExample, numOfLabels)

cutoff = int(numOfExamples*0.7)

for i in range(0, cutoff):
   trainData.addSample(X[i,:], Y[i,:])

# setting the field names
trainData.setField('input', X[0:cutoff, :])
trainData.setField('target', Y[0:cutoff, :])

for i in range(cutoff, numOfExamples):
   testData.addSample(X[i,:], Y[i,:])

testData.setField('input', X[cutoff:numOfExamples, :])
testData.setField('target', Y[cutoff:numOfExamples, :])

# ============= defining the layers of the network ==============
    
inputLayerSize = sizeOfExample
hiddenLayerSize0 = sizeOfExample
outputLayerSize = numOfLabels
    
inputLayer = LinearLayer(inputLayerSize)
hiddenLayer0 = SigmoidLayer(hiddenLayerSize0)
outputLayer = SoftmaxLayer(outputLayerSize)
      
# ============= building the feedforward network ================
    
ffNetwork = FeedForwardNetwork() 
    
# adding the layers to the network
ffNetwork.addInputModule(inputLayer)
ffNetwork.addModule(hiddenLayer0)
ffNetwork.addOutputModule(outputLayer)
    
# initializing the thetas
theta1 = FullConnection(inputLayer, hiddenLayer0)
theta2 = FullConnection(hiddenLayer0, outputLayer)
    
# connecting the layers with thetas
ffNetwork.addConnection(theta1)
ffNetwork.addConnection(theta2)
    
# this sets the network
# input_layer->theta1->hidden_layer->theta2->output_layer
ffNetwork.sortModules()

# ============== basic testing =======================

print('_______testing without training________')

testIndex = randomIndex
testInput = X[testIndex, :]

prediction = ffNetwork.activate(testInput)
p = argmax(prediction, axis=0)

#plotData(X[:, 0:sizeOfExample-1], Y, randomIndex)
print("predicted output is" , p)

# ============= building the backpropogation network ================

print('_______testing after training_______')   

trueTrain = trainData['target'].argmax(axis=1)
trueTest = testData['target'].argmax(axis=1)

EPOCHS = 35
backPropTrainer = BackpropTrainer(ffNetwork, dataset=trainData, verbose = True)
for i in range(EPOCHS):
    backPropTrainer.trainEpochs(1)

# calculatig the error percentage
outputTrain = ffNetwork.activateOnDataset(trainData)
outputTrain = outputTrain.argmax(axis=1)
trainResult = percentError(outputTrain, trueTrain)

outputTest = ffNetwork.activateOnDataset(testData)
outputTest = outputTest.argmax(axis=1)
testResult = percentError(outputTest, trueTest)

print('training set accuracy:', 100 - trainResult, 'test set accuracy:', 100 - testResult)



