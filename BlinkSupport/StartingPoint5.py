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
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeRawPixels=False, includeIntensities=False)
yTrain = yTrainRaw
yTest = yTestRaw

######
#import MostCommonModel
#model = MostCommonModel.MostCommonModel()
#model.fit(xTrain, yTrain)
#yTestPredicted = model.predict(xTest)
#print("Most Common Accuracy:", EvaluationsStub.Accuracy(yTest, yTestPredicted))
#EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

######
#import DecisionTreeModel
#model = DecisionTreeModel.DecisionTreeModel()
#model.fit(xTrain, yTrain, minToSplit=50)
#yTestPredicted = model.predict(xTest)
#print("Decision Tree Accuracy:", EvaluationsStub.Accuracy(yTest, yTestPredicted))
#EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

######
import RandomForestsModel
model = RandomForestsModel.RandomForestsModel()

#min to split, num trees and feature restriction).
numTreesToTry = [5]

for tryval in numTreesToTry:
    model.fit(xTrain, yTrain, numTrees=tryval, minSplit=15, useBagging=True, featureRestriction=20)
    
    thresholdval = -0.01
    for j in range(101):
        thresholdval += 0.01
        yTestPredicted = model.predictThres(xTest, thresholdval)

        print(thresholdval, EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted), EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted))

    yTestPredicted = model.predictThres(xTest)
    print(tryval, " TREES - Random Forests Accuracy: ", EvaluationsStub.Accuracy(yTest, yTestPredicted))
    EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

minSplitToTry = [10, 20, 35, 50]

for tryval in minSplitToTry:
    
    model.fit(xTrain, yTrain, numTrees=15, minSplit=tryval, useBagging=True, featureRestriction=0)

    #thresholdval = -0.01
    #for j in range(101):
    #    thresholdval += 0.01
    #    yTestPredicted = model.predictThres(xTest, thresholdval)

    #    print(thresholdval, EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted), EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted))


    yTestPredicted = model.predictThres(xTest)
    print(tryval, " MINSPLIT - Random Forests Accuracy: ", EvaluationsStub.Accuracy(yTest, yTestPredicted))
    #EvaluationsStub.ExecuteAll(yTest, yTestPredicted)


frToTry = [5, 10, 15, 20]

for tryval in frToTry:
    model.fit(xTrain, yTrain, numTrees=10, minSplit=10, useBagging=True, featureRestriction=tryval)


    yTestPredicted = model.predictThres(xTest)
    print(tryval, " FEATURE RESTRICTION - Random Forests Accuracy: ", EvaluationsStub.Accuracy(yTest, yTestPredicted))
    #EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

##### for visualizing in 2d
#for i in range(500):
#    print("%f, %f, %d" % (xTrain[i][0], xTrain[i][1], yTrain[i]))

##### sample image debugging output

#import PIL
#from PIL import Image

#i = Image.open(xTrainRaw[1])
##i.save("..\\..\\..\\Datasets\\FaceData\\test.jpg")

#print(i.format, i.size)

## Sobel operator
#xEdges = Assignment5Support.Convolution3x3(i, [[1, 2, 1],[0,0,0],[-1,-2,-1]])
#yEdges = Assignment5Support.Convolution3x3(i, [[1, 0, -1],[2,0,-2],[1,0,-1]])

#pixels = i.load()

#for x in range(i.size[0]):
#    for y in range(i.size[1]):
#        pixels[x,y] = abs(xEdges[x][y])

#i.save("c:\\Users\\ghult\\Desktop\\testEdgesY.jpg")