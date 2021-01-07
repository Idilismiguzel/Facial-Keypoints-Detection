## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.conv1 = nn.Conv2d(1, 32, 5) # output tensor = (W-F)/S + 1 = (96-5)/1 +1 = 92
        self.conv2 = nn.Conv2d(32, 64, 3) # output tensor = (W-F)/S + 1 = (46-3)/1 +1 = 42 
        self.conv3 = nn.Conv2d(64, 128, 2) # output tensor = (W-F)/S + 1 = (21-2)/1 +1 = 20
        self.conv4 = nn.Conv2d(128, 256, 1) # output tensor = (W-F)/S + 1 = (10-1)/1 +1 = 10
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout Layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64*10*10, 1000)
        self.fc2 = nn.Linear(1000, 136)
        #self.fc3 = nn.Linear(500, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout4(x)
        
        # Flattening before passing to the fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout5(x)
        
        x = self.fc2(x)
#         x = F.relu(x)
        
#         x = self.dropout6(x)
#         x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
