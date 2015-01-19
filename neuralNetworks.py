from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, SoftmaxLayer, FullConnection

from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet

from random import randint
from pylab import imshow
from scipy import io
from numpy import unique, c_, ones

import matplotlib.pyplot as plt


def plotData(X, Y, c):
    '''plots the input data '''

    # plot any one case (20x20 image) from the input
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    inputImg = X[c,:]
    imshow((inputImg.reshape(20,20)).T, cmap = 'Greys')
    
    #plot the same ouptut case
    print('the digit printed is', Y[c])   



# ============= load the data================
    
data = io.loadmat('ex4data1.mat')
    
# making X and Y numpy arrays
X = data['X']
Y = data['y']

# changing identity of digit 0 from 10 to 0 in array
Y[Y==10] = 0
  
# ============= get no of ouputs possible ================

# getting the no. of different classes in the output    
numOfLabels = unique(Y).size    
    
# ============= plotting ================

print('plotting a random digit from the input')
c = randint(0, X.shape[0])
plotData(X, Y, c)
plt.show()
    
# ============= building the dataset ================

X = c_[ones(X.shape[0]), X]
numOfExamples, sizeOfExample = X.shape

allData = ClassificationDataSet(sizeOfExample, numOfLabels)
    
for i in range(0, numOfExamples):
    allData.addSample(X[i,:], Y[i,:])

# separating training and test dataset
trainData, testData = allData.splitWithProportion(0.78)

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
    
# this sets the network i.e input_layer->theta1->hidden_layer->theta2->output_layer
ffNetwork.sortModules()
    
# ============= building the backpropogation network ================
    
backPropTrainer = BackpropTrainer(ffNetwork, dataset=trainData, verbose = True)
backPropTrainer.trainUntilConvergence()

# calculatig the error percentage
trainResult = percentError(backPropTrainer.testOnClassData(), trainData['target'])
testResult = percentError(backPropTrainer.testOnClassData(testData), testData['target'])

print('training set accuracy:', 100-trainResult*100, 'test set accuracy:', 100-testResult*100)


