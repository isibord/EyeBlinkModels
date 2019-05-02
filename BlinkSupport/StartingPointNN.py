## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support
import EvaluationsStub

## NOTE update this with your equivalent code..
#import TrainTestSplit

kDataPath = "dataset_B_Eye_Images"

(xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

print("Calculating features...")
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=False, includeRawPixels=False, includeIntensities=True)
yTrain = yTrainRaw
yTest = yTestRaw


######
import NeuralNetworkModel
model = NeuralNetworkModel.NeuralNetworkModel()

#Train models with every combination of hidden layer in [1, 2] 
#and hidden nodes per layer in [ 2, 5, 10, 15, 20 ]. For each, use 200 iterations with step size=0.05
model.fit(xTrain, yTrain, xTrainRaw)
  
#Produce a plot with one line for each of these run with the iteration number on the x axis and training set loss on the y axis.


#Produce a separate plot with one line for each of these runs but with the test set losses on the y axis.