import torch

class SimpleBlinkNeuralNetworkLeNet(torch.nn.Module):
    def __init__(self, hiddenNodes = 20):
        super(SimpleBlinkNeuralNetworkLeNet, self).__init__()

        # This will reduce image to 20 x 20
        self.convolution1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)

        #This will reduce image to 10 x 10
        self.pooling1 = torch.nn.AvgPool2d(kernel_size = 2, stride = 2) 

        #This will reduce image to 6 x 6
        self.convolution2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        #This will reduce image to 3 x 3
        self.pooling2 = torch.nn.AvgPool2d(kernel_size = 2, stride=2)

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(16*3*3, 120),
           torch.nn.Sigmoid()
           )

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedTwo = torch.nn.Sequential(
           torch.nn.Linear(120, 84),
           torch.nn.Sigmoid()
           )

        # Fully connected layer from the hidden layer to a sin gle output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(84, 1),
            torch.nn.Sigmoid()
            )

    def forward(self, x):
        # Apply the layers created at initialization time in order
        out_c =  x
        out = self.convolution1(x)
        out = self.pooling1(out)
        #out = out.reshape(out.size(0), -1)
        out = self.convolution2(out)
        out = self.pooling2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fullyConnectedOne(out)
        out = self.fullyConnectedTwo(out)
        out = self.outputLayer(out)

        return out