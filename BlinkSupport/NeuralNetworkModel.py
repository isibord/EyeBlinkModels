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



import numpy as np

#Input array
#X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
X = np.array(xTrain)

#Output
#y=np.array([[1],[1],[0]])
y = np.array([[item] for item in yTrain])

#Sigmoid Function

def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

def sum_squared_error(outputs, targets, derivative=False ):
    #(Equation 4.2 in Mitchell). E() = 1/2 sum_sample (y^-y)^2
    if derivative:
        return targets - outputs 
    else:
        #return 0.5 * np.mean(np.sum( np.power(targets - outputs, 2), axis = 1 ))
        #return 0.5 * np.sum(np.power(outputs - targets, 2))
        diff = outputs - targets
        eachloss =  0.5 * np.power(diff, 2)
        return np.mean(eachloss)

def errorterm(outputs, targets):
    #𝛿_𝑜=𝑦^^ (1−𝑦^^)(𝑦−𝑦^^)
    return outputs * (1 - outputs) * (targets - outputs)


#Variable initialization
epoch=200 #Setting training iterations
lr=0.05 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 5 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(low=-0.05, high=0.05, size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(low=-0.05, high=0.05, size=(1,hiddenlayer_neurons))
wout=np.random.uniform(low=-0.05, high=0.05, size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(low=-0.05, high=0.05, size=(1,output_neurons))


for i in range(epoch):

    #Forward Propogation
    hidden_layer_input_beforebias = np.dot(X,wh)
    hidden_layer_input = hidden_layer_input_beforebias + bh

    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1 + bout
    output = sigmoid(output_layer_input)

    #Backpropagation
    E = y-output
    #E = sum_squared_error(output, y, True)
    #ET = errorterm(output, y)

    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)

    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0,keepdims=True) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * lr

    print(i+1, sum_squared_error(output, y, False))

#print(output)
#print(wh)
#print(bh)
#print(wout)
#print(bout)