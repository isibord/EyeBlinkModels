import os
import random
import numpy as np
import math

def LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True, shuffle=True):
    xRaw = []
    yRaw = []
    
    if includeLeftEye:
        closedEyeDir = os.path.join(kDataPath, "closedLeftEyes")
        for fileName in os.listdir(closedEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(closedEyeDir, fileName))
                yRaw.append(1)

        openEyeDir = os.path.join(kDataPath, "openLeftEyes")
        for fileName in os.listdir(openEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(openEyeDir, fileName))
                yRaw.append(0)

    if includeRightEye:
        closedEyeDir = os.path.join(kDataPath, "closedRightEyes")
        for fileName in os.listdir(closedEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(closedEyeDir, fileName))
                yRaw.append(1)

        openEyeDir = os.path.join(kDataPath, "openRightEyes")
        for fileName in os.listdir(openEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(openEyeDir, fileName))
                yRaw.append(0)

    if shuffle:
        random.seed(1000)

        index = [i for i in range(len(xRaw))]
        random.shuffle(index)

        xOrig = xRaw
        xRaw = []

        yOrig = yRaw
        yRaw = []

        for i in index:
            xRaw.append(xOrig[i])
            yRaw.append(yOrig[i])

    return (xRaw, yRaw)

def TrainTestSplit(x, y, percentTest = .25):
    if(len(x) != len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xTrain = x[numTest:]
    yTest = y[:numTest]
    yTrain = y[numTest:]

    return (xTrain, yTrain, xTest, yTest)

from PIL import Image

def Convolution3x3(image, filter):
    # check that the filter is formated correctly
    if not (len(filter) == 3 and len(filter[0]) == 3 and len(filter[1]) == 3 and len(filter[2]) == 3):
        raise UserWarning("Filter is not formatted correctly, should be [[x,x,x], [x,x,x], [x,x,x]]")

    xSize = image.size[0]
    ySize = image.size[1]
    pixels = image.load()

    answer = []
    for x in range(xSize):
        answer.append([ 0 for y in range(ySize) ])

    # skip the edges
    for x in range(1, xSize - 1):
        for y in range(1, ySize - 1):
            value = 0

            for filterX in range(len(filter)):
                for filterY in range(len(filter)):
                    imageX = x + (filterX - 1)
                    imageY = y + (filterY - 1)

                    #value += pixels[imageX, imageY] * filter[filterX][filterY]
                    value += (pixels[imageX, imageY]/255.0) * filter[filterX][filterY]

            answer[x][y] = value

    return answer

def CalculateHistogramFeatures(yEdges, xSize, ySize, numPixels):
    features = []
    HistY02 = 0
    HistY24 = 0
    HistY46 = 0
    HistY68 = 0
    HistY81 = 0

    for x in range(xSize):
        for y in range(ySize):
            if abs(yEdges[x][y]) <= 0.2:
                HistY02 += 1
            elif abs(yEdges[x][y]) <= 0.4:
                HistY24 +=1
            elif abs(yEdges[x][y]) <= 0.6:
                HistY46 += 1
            elif abs(yEdges[x][y]) <= 0.8:
                HistY68 += 1
            else:
                HistY81 += 1
            
    features.append(HistY02 / numPixels)
    features.append(HistY24 / numPixels)
    features.append(HistY46 / numPixels)
    features.append(HistY68 / numPixels)
    features.append(HistY81 / numPixels)

    return features

def CalculateGradientFeatures(yEdges):
    features = []
    #Split into MXN images
    im =  np.asarray(yEdges)
    M = 8
    N = 8
    tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    for tile in tiles:
        max = tile.max()
        min = tile.min()
        sumGradient = sum([sum([abs(value) for value in row]) for row in tile])
        count = sum([len(row) for row in tile])
        features.append(sumGradient / count)
        features.append(max)
        features.append(min)

    return features


def Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeRawPixels=False, includeIntensities=False):
    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for sample in xTrainRaw:
        features = []

        image = Image.open(sample)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if includeGradients:
            # average Y gradient strength
            yEdges = Convolution3x3(image, [[1, 2, 1],[0,0,0],[-1,-2,-1]])
                                 
            sumGradient = sum([sum([abs(value) for value in row]) for row in yEdges])
            count = sum([len(row) for row in yEdges])

            features.append(sumGradient / count)

            # average Y gradient strength in middle 3rd
            sumGradient = sum([sum([abs(value) for value in row[8:16]]) for row in yEdges])
            count = sum([len(row[8:16]) for row in yEdges])

            features.append(sumGradient / count)
            
            ##ASSIGNMENT ADD##

            #xEdges = Convolution3x3(image, [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

            #features.extend(CalculateHistogramFeatures(yEdges, xSize, ySize, numPixels))
            #features.extend(CalculateGradientFeatures(yEdges))
            
            #features.extend(CalculateHistogramFeatures(xEdges, xSize, ySize, numPixels))
            #features.extend(CalculateGradientFeatures(xEdges))
            

        if includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    features.append(pixels[x,y])


        if includeIntensities:
            for x in range(0, xSize, 2):
                for y in range(0, ySize, 2):
                    features.append(pixels[x,y]/255.0)

        xTrain.append(features)

    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for sample in xTestRaw:
        features = []
        
        image = Image.open(sample)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if includeGradients:
            # average Y gradient strength
            yEdges = Convolution3x3(image, [[1, 2, 1],[0,0,0],[-1,-2,-1]])
            
            sumGradient = sum([sum([abs(value) for value in row]) for row in yEdges])
            count = sum([len(row) for row in yEdges])

            features.append(sumGradient / count)

            # average Y gradient strength in middle 3rd
            sumGradient = sum([sum([abs(value) for value in row[8:16]]) for row in yEdges])
            count = sum([len(row[8:16]) for row in yEdges])

            features.append(sumGradient / count)

            ##ASSIGNMENT ADD##

            #xEdges = Convolution3x3(image, [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

            #features.extend(CalculateHistogramFeatures(yEdges, xSize, ySize, numPixels))
            #features.extend(CalculateGradientFeatures(yEdges))
            
            #features.extend(CalculateHistogramFeatures(xEdges, xSize, ySize, numPixels))
            #features.extend(CalculateGradientFeatures(xEdges))

        if includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    features.append(pixels[x,y])

        if includeIntensities:
            for x in range(0, xSize, 2):
                for y in range(0, ySize, 2):
                    features.append(pixels[x,y]/255.0)

        xTest.append(features)

    return (xTrain, xTest)


import PIL
from PIL import Image

def VisualizeWeights(weightArray, outputPath):
    size = 12

    # note the extra weight for the bias is where the +1 comes from, just ignore it
    if len(weightArray) != (size*size) + 1:
        raise UserWarning("size of the weight array is %d but it should be %d" % (len(weightArray), (size*size) + 1))

    if not outputPath.endswith(".jpg"):
        raise UserWarning("output path should be a path to a file that ends in .jpg, it is currently: %s" % (outputPath))

    image = Image.new("L", (size,size))

    pixels = image.load()

    for x in range(size):
        for y in range(size):
            pixels[x,y] = int(abs(weightArray[(x*size) + y]) * 255)

    image.save(outputPath)

def GetAllDataExceptFold(xTrain, yTrain, i, numFolds):
    xTrainSplit = partition(xTrain, numFolds)
    yTrainSplit = partition(yTrain, numFolds)

    xTrainExceptFold = []
    yTrainExceptFold = []
    for num in range(numFolds):
        if num != i:
            xTrainExceptFold.extend(xTrainSplit[num])
            yTrainExceptFold.extend(yTrainSplit[num])
   
    return (xTrainExceptFold, yTrainExceptFold)


def GetDataInFold(xTrain, yTrain, i, numFolds):
    return (partition(xTrain, numFolds)[i], partition(yTrain, numFolds)[i])


def partition(seq, chunks):
    """Splits the sequence into equal sized chunks and them as a list"""
    result = []
    for i in range(chunks):
        chunk = []
        for element in seq[i:len(seq):chunks]:
            chunk.append(element)
        result.append(chunk)
    return result