import numpy as np

class KNNModel(object):
    """A KNN Model"""
    
    def __init__(self):
        pass
        
    def fit(self, x, y, k):
        self.trainx = x
        self.trainy = y
        self.kval = k


    def predictThres(self, x, batchThres=True, threshold=0.5):
        # for every sample in training set
        # compute the distance to xText (using l2 norm)

        probEst = []

        for eachx in x:
            distDict = {}
            eachx = np.asarray(eachx)
            for i in range(len(self.trainx)):
                trainxval = np.asarray(self.trainx[i])
                normval = np.linalg.norm(trainxval-eachx)
                distDict[i] = normval

            sorted_x = sorted(distDict.items(), key=lambda kv: kv[1])
            count1s = 0
            for j in range(self.kval):
               (idx, norm) = sorted_x[j]
               yVal = self.trainy[idx]
               if yVal == 1:
                   count1s +=1
            probEst.append(count1s/self.kval)

        predictionList = []

        if batchThres:
            thresholdval = -0.01
            for j in range(101):
                predictions = []
                thresholdval += 0.01
                for probVal in probEst:
                    predictions.append(1 if probVal > thresholdval else 0)
                predictionList.append(predictions)
        else:
            predictions = []

            for probVal in probEst:
                predictions.append(1 if probVal > threshold else 0)
            predictionList.append(predictions)


        return predictionList
