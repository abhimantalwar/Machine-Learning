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
    '''divides the data into training, cross validation and test datsets '''
    
    # get input matrix shape and no. of output classes
    n, m = X.shape
    k = unique(Y).size

    # examples per output class
    epc = int(n/k)

    # get the train, cross validation and test examples per output class
    epcTrain = int(epc*cutoff)
    epcCV = int((epc-epcTrain)/2)
    epcTest = epc - epcTrain - epcCV
    
    # choosing training, cross validation and test dataset size
    trainDataSize = n*cutoff
    cvDataSize = int((n - trainDataSize)/2)
    testDataSize = n - trainDataSize - cvDataSize
    
    # initializing training dataset
    XTrain = zeros((trainDataSize, m))
    YTrain = zeros(trainDataSize)

    # initializing cross validation dataset
    XCV = zeros((cvDataSize, m))
    YCV = zeros(cvDataSize)

    # initializing test dataset
    XTest = zeros((testDataSize , m))
    YTest = zeros(testDataSize)

    # assigning exampples to training dataset
    for i in range(0,k):
        for j in range(0, epcTrain):
            XTrain[i*epcTrain + j] = X[i*epc + j]
            YTrain[i*epcTrain + j] = Y[i*epc + j]

    # assigning exampples to cross validation dataset
    for i in range(0,k):
        for j in range(0, epcCV):
            XCV[i*epcCV + j] = X[i*epc + j+epcTrain]
            YCV[i*epcCV + j] = Y[i*epc + j+epcTrain]
    

    # assigning exampples to test dataset
    for i in range(0,k):
        for j in range(0, epcTest):
            XTest[i*epcTest + j] = X[i*epc + j+epcTrain+epcCV]
            YTest[i*epcTest + j] = Y[i*epc + j+epcTrain+epcCV]

    return XTrain, YTrain, XCV, YCV, XTest, YTest

#=========================== arrange data for use ==========================

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

# choosing training , cross validation and test dataset size
cutoff = 0.60
XTrain, YTrain, XCV, YCV, XTest, YTest = arrangeData(X, Y, cutoff)

# plotting 
print('plotting a random digit from the input')
randomIndex = randint(0, X.shape[0])
plotData(X, Y, randomIndex)
plt.show()

# ================== set params for plotting graph =========================

# setting the gamma and C range
gammaRange = 10
CRange = 10

# gamma and C values
# change it acc to your data
gammaStart = 0.05
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

# ====================== training and testing =============================

# let the looping begin 
for g in range(0, gammaRange):
    for c in range(0, CRange):

        #creating a Support vector classifier
        classifier = svm.SVC(C = CArray[c], kernel='rbf', gamma = GammaArray[g], tol=0.001)
         
        # learning on the training dataset
        classifier.fit(XTrain, YTrain)

        # predicting on the cross validation dataset
        prediction = classifier.predict(XCV)
        eff[g, c] = 100 * metrics.accuracy_score(YCV, prediction, normalize = True)
        print(eff[g, c])

# ================== post training and testing work ========================

# getting C and gamma for max effciency
gammaMax, CMax = unravel_index(argmax(eff), eff.shape)
effMax = eff[gammaMax, CMax]
print('max eff is', effMax , 'and it comes at gamma = ', GammaArray[gammaMax], 'and C =', CArray[CMax] )

# now using the best params got from CV on test dataset
# and getting efficiency for test dataset
classifier = svm.SVC(C = CArray[CMax], kernel='rbf', gamma = GammaArray[gammaMax], tol=0.001)
classifier.fit(XTrain, YTrain)
prediction = classifier.predict(XTest)
print('efficiency on test dataset is,', metrics.accuracy_score(YCV, prediction, normalize = True))

# testing on a random input
print('testing on a random input')
c = randint(0, XTest.shape[0])
predictedOutput = classifier.predict(XTest[c])[0]
actualOutput = YTest[c]
print('predicted output is', predictedOutput, 'and actual output is', actualOutput)

#========================== plotting the 3d graph ================================

# plotting the graph
fig = plt.figure()
Ax = fig.add_subplot(111,projection = '3d')

xGraph, yGraph = meshgrid(GammaArray, CArray)
Ax.plot_surface(xGraph, yGraph, eff, rstride=1, cstride=1)#, cmap = 'hot')
plt.show()

