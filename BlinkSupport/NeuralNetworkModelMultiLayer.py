import Assignment5Support
import EvaluationsStub
import numpy as np


kDataPath = "dataset_B_Eye_Images"

(xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

print("Calculating features...")
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=False, includeRawPixels=False, includeIntensities=True)
yTrain = yTrainRaw
yTest = yTestRaw


np.random.seed(0)

X = np.array(xTrain)
y = np.array([[item] for item in yTrain])

def sigmoid(x):
    return 1.0 /(1.0 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1.0 - x)

def sum_squared_error(outputs, targets, derivative=False ):
    #(Equation 4.2 in Mitchell). E() = 1/2 sum_sample (y^-y)^2
    if derivative:
        return targets - outputs 
    else:
        #return 0.5 * np.mean(np.sum( np.power(targets - outputs, 2), axis = 1 ))
        #return 0.5 * np.sum(np.power(outputs - targets, 2))
        diff = targets - outputs
        eachloss =  0.5 * np.power(diff, 2)
        return np.mean(eachloss)

def errorterm(outputs, targets):
    #𝛿_𝑜=𝑦^^ (1−𝑦^^)(𝑦−𝑦^^)
    return outputs * (1.0 - outputs) * (targets - outputs)


numIteration = 200
learning_rate = 0.05
inputlayer_neurons = X.shape[1] #number of features in data set
output_neurons = 1 #number of neurons at output layer


layerNeurons = [ 2, 5, 10, 15, 20 ]

for ln in layerNeurons:
    hiddenlayer_neurons = ln #number of hidden layers neurons
    print("##### ", ln)

    #weight and bias initialization
    wh=np.random.uniform(low=0, high=0.001, size=(inputlayer_neurons,hiddenlayer_neurons))
    bh=np.random.uniform(low=0, high=0.001, size=(1,hiddenlayer_neurons))
    wout=np.random.uniform(low=0, high=0.005, size=(hiddenlayer_neurons,output_neurons))
    bout=np.random.uniform(low=0, high=0.005, size=(1,output_neurons))


    for i in range(numIteration):

        #Forward Propogation
        hidden_layer_input_beforebias = np.dot(X,wh)
        hidden_layer_input = hidden_layer_input_beforebias + bh

        hiddenlayer_activations = sigmoid(hidden_layer_input)

        output_layer_input_beforebias = np.dot(hiddenlayer_activations, wout)
        output_layer_input = output_layer_input_beforebias + bout

        output = sigmoid(output_layer_input)

        #Backpropagation
        #E = y – output
        E = y-output

        #ET = errorterm(output, y)

        slope_output_layer = derivatives_sigmoid(output)
        d_output = E * slope_output_layer

        wout += hiddenlayer_activations.T.dot(d_output) * learning_rate
        bout += np.sum(d_output, axis=0,keepdims=True) * learning_rate

        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)

        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wh += X.T.dot(d_hiddenlayer) * learning_rate
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * learning_rate

        #print(i+1, E)
        print(i+1, sum_squared_error(output, y, False))
        #print(i+1, ET)
        #print(i+1, d_output)

    predictions = []

    for probVal in output:
        predictions.append(1 if probVal[0] >= 0.5 else 0)

    print(EvaluationsStub.Accuracy(yTrain, predictions))


#print(output)
#print(wh)
#print(bh)
#print(wout)
#print(bout)