## Some of this references my answers to previous assignments.

import Assignment5Support
import EvaluationsStub


## NOTE update this with your equivalent code.

kDataPath = "dataset_B_Eye_Images"

(xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

from PIL import Image
import torchvision.transforms as transforms
import torch 

(foldTrainX, foldTrainY)  = Assignment5Support.GetAllDataExceptFold(xTrainRaw, yTrainRaw, 2, 5)
(foldValidationX, foldValidationY) = Assignment5Support.GetDataInFold(xTrainRaw, yTrainRaw, 2, 5)

# Load the images and then convert them into tensors (no normalization)
xTrainImages = [ Image.open(path) for path in foldTrainX ]
xTrainFlip = [ transforms.ToTensor()(image.transpose(Image.FLIP_LEFT_RIGHT)) for image in xTrainImages ]
xTrainReg = [ transforms.ToTensor()(image) for image in xTrainImages ]
xTrainReg.extend(xTrainFlip)
xTrain = torch.stack(xTrainReg)

xValidationImages = [ Image.open(path) for path in foldValidationX ]
xValidation = torch.stack([ transforms.ToTensor()(image) for image in xValidationImages ])

xTestImages = [ Image.open(path) for path in xTestRaw ]
xTest = torch.stack([ transforms.ToTensor()(image) for image in xTestImages ])

foldTrainY.extend(foldTrainY)
yTrain = torch.Tensor([ [ yValue ] for yValue in foldTrainY ])
yValidation = torch.Tensor([ [ yValue ] for yValue in foldValidationY ])
yTest = torch.Tensor([ [ yValue ] for yValue in yTestRaw ])


######
######

import SimpleBlinkNeuralNetworkLeNetMod

# Create the model and set up:
#     the loss function to use (Mean Square Error)
#     the optimization method (Stochastic Gradient Descent) and the step size
model = SimpleBlinkNeuralNetworkLeNetMod.SimpleBlinkNeuralNetworkLeNet()
lossFunction = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.95)

for i in range(300):
    # Do the forward pass
    yTrainPredicted = model(xTrain)

    # Compute the training set loss
    loss = lossFunction(yTrainPredicted, yTrain)
    print(i, loss.item())
    
    # Reset the gradients in the network to zero
    optimizer.zero_grad()

    # Backprop the errors from the loss on this iteration
    loss.backward()

    # Do a weight update step
    optimizer.step()

    
yValidationPred = model(xValidation)

yPred = [ 1 if pred > 0.5 else 0 for pred in yValidationPred ]

print("Accuracy on holdout set:", EvaluationsStub.Accuracy(foldValidationY, yPred))


yTestPredicted = model(xTest)

yPred = [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ]

print("Accuracy on Test Set:", EvaluationsStub.Accuracy(yTest, yPred))
EvaluationsStub.ExecuteAll(yTest, yPred)