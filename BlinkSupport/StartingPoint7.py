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
import KMeansModel
model = KMeansModel.KMeansModel()


model.fit(xTrain, yTrain, xTrainRaw)
   
#yTestPredictedList = model.predictThres(xTest, True)
    
#thresholdval = -0.01
#midThresPred = []
#for yTestPredicted in yTestPredictedList:
#    thresholdval += 0.01
#    print(EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted), EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted))
#    if thresholdval > 0.49 and thresholdval < 0.51:
#        midThresPred = yTestPredicted

#print(tryval, " K vals - KNN Accuracy: ", EvaluationsStub.Accuracy(yTest, midThresPred))
#EvaluationsStub.ExecuteAll(yTest, midThresPred)


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