from sklearn import svm, metrics

from random import shuffle, randint
from pylab import imshow
from scipy import io
from numpy import unique, zeros, hstack, meshgrid, unravel_index, argmax

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotData(X, Y, c):
    '''plots the input data '''

    # plot any one case (20x20 image) from the input
    # the image matrix will have to be transposed to be viewed correcty
    inputImg = X[c,:]
    imshow((inputImg.reshape(20,20)).T, cmap = 'Greys')
    
    #plot the same ouptut case
    print('the digit printed is', Y[c][0])


def arrangeData(X, Y, cutoff):
    '''divides the data into training and test datsets '''
    
    # get input matrix shape and no. of output classes
    n, m = X.shape
    k = unique(Y).size

    # examples per output class
    epc = int(n/k)

    # get the train and test examples per output class
    epcTrain = int(epc*cutoff)
    epcTest = epc - epcTrain
    
    # choosing training and test dataset size
    trainDataSize = n*cutoff
    testDataSize = n - trainDataSize

    # initializing
    XTrain = zeros((trainDataSize, m))
    YTrain = zeros(trainDataSize)

    XTest = zeros((testDataSize , m))
    YTest = zeros(testDataSize)

    # assigning
    for i in range(0,k):
        for j in range(0, epcTrain):
            XTrain[i*epcTrain + j] = X[i*epc + j]
            YTrain[i*epcTrain + j] = Y[i*epc + j]

    for i in range(0,k):
        for j in range(0, epcTest):
            XTest[i*epcTest + j] = X[i*epc + j+epcTrain]
            YTest[i*epcTest + j] = Y[i*epc + j+epcTrain]

    return XTrain, YTrain, XTest, YTest


# load the data
data = io.loadmat('ex4data1.mat')

# making X and Y numpy arrays
X = data['X']
Y = data['y']

# changing identity of digit '0' from 10 to 0 in array
Y[Y==10] = 0

# getting the no. of examples and size of each example
numOfExamples, sizeOfExample = X.shape

# getting the no. of different classes in the output    
numOfLabels = unique(Y).size   

# choosing training and test dataset size
cutoff = 0.70
XTrain, YTrain, XTest, YTest = arrangeData(X, Y, cutoff)

# plotting 
print('plotting a random digit from the input')
randomIndex = randint(0, X.shape[0])
plotData(X, Y, randomIndex)
plt.show()

# ================== plot graph to get best params =========================

# setting the gamma and C range
gammaRange = 2
CRange = 2

# how gamma and C will vary
# change it acc to your data
gammaStart = 0.005
gammaIncrement = 0.005
CStart = 10
CIncrement = 10

# initializing x, y and z arrays for plotting
GammaArray = zeros(gammaRange)
CArray = zeros(CRange)
eff = zeros((gammaRange, CRange))

# gamma values to iterate over
for i in range(0,gammaRange):
    GammaArray[i] = gammaStart + gammaIncrement*i

# C values to iterate over
for i in range(0,CRange):
    CArray[i] = CStart + CIncrement*i

# let the looping begin 
for g in range(0, gammaRange):
    for c in range(0, CRange):

        #creating a Support vector classifier
        classifier = svm.SVC(C = CArray[c], kernel='rbf', gamma = GammaArray[g], tol=0.001)
         
        # learning on the digits
        classifier.fit(XTrain, YTrain)

        # predicting on the learned/trained data
        prediction = classifier.predict(XTest)

        '''
        # printing the outputs
        print(metrics.classification_report(YTest, prediction))
        print(metrics.confusion_matrix(YTest, prediction))
        print( 'C=', c, 'gamma=', g/1000, 'and efficiency=', metrics.accuracy_score(YTest, prediction, normalize = True))
        '''
        
        print(metrics.accuracy_score(YTest, prediction, normalize = True))

        # getting efficiency
        eff[g, c] = 100 * metrics.accuracy_score(YTest, prediction, normalize = True)

# getting C and gamma for max effciency
gammaMax, CMax = unravel_index(argmax(eff), eff.shape)
effMax = eff[gammaMax, CMax]

print('max eff is', effMax , 'and it comes at gamma = ', GammaArray[gammaMax], 'and C =', CArray[CMax] )
c = randint(0, XTest.shape[0])
predictedOutput = classifier.predict(XTest[c])[0]
actualOutput = YTest[c]

print('testing on a random input')
print('predicted output is', predictedOutput, 'and actual output is', actualOutput)

fig = plt.figure()
Ax = Axes3D(fig)

Ax.plot_surface(GammaArray, CArray, eff, rstride=1, cstride=1)#, cmap = 'hot')
plt.show()
