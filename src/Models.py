import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as T
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from vizdoom import *


class Model(nn.Module):
    def __init__(self, device="cpu"):
        super(Model, self).__init__()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def forward(self, x):
        raise NotImplementedError

    def set_device(self, device):
        self.device = device

    def predict(self, x):
        raise NotImplementedError


class SimpleLinearNN(Model):
    def __init__(self, x_size, y_size, action_space, dim=3):
        super(SimpleLinearNN, self).__init__()
        rescale_factor = 0.2
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_size = (self.xs * self.ys) * dim
        self.dense1 = nn.Linear(img_size, 100)
        self.dense2 = nn.Linear(100, action_space)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = x.to(self.device)
        x = T.resize(x, size=[self.xs, self.ys])
        x = torch.flatten(x)
        #x = self.conv1(x)
        #x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        print(x)
        return torch.argmax(x)


class SimpleConvNN(Model):
    def __init__(self, x_size, y_size, action_space, dim=3):
        super(SimpleConvNN, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        ks = 3
        self.conv1 = nn.Conv2d(dim, 64, ks)
        self.x_size -= 2
        self.y_size -= 2
        self.conv2 = nn.Conv2d(64, 1, ks)
        self.x_size -= 2
        self.y_size -= 2
        self.dense1 = nn.Linear(self.x_size * self.y_size, action_space)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = x.to(self.device)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.flatten(x)
        #x = x.
        #x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        #x = x[0]
        return x

    def predict(self, x):
        x = self.forward(x)
        print(x)
        return torch.argmax(x)


class ConvLinearNN(Model):
    def __init__(self, x_size, y_size, action_space, dim=3):
        super(ConvLinearNN, self).__init__()
        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        ks = 3
        img_x = self.xs
        img_y = self.ys
        self.blur_kernel = int(1/rescale_factor)
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.conv1 = nn.Conv2d(1, 8, ks)
        img_x -= (ks-1)
        img_y -= (ks-1)
        ks = 3
        self.conv2 = nn.Conv2d(8, 1, ks)
        img_x -= (ks-1)
        img_y -= (ks-1)

        img_size = (img_x * img_y)
        self.dense1 = nn.Linear(img_size, 100)
        self.dense2 = nn.Linear(100, action_space)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = x.to(self.device)

        x = T.resize(x, size=(self.xs, self.ys))
        x = T.rgb_to_grayscale(x)

        x = self.conv1(x) # convolutional
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = torch.flatten(x) # Linear
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        print(x)
        #print(x)
        return torch.argmax(x)

# Image size recommended:
# større end 8 eller sådan kanaler, f.eks. 32 og 64
# kernel størrelse ikke over 10 og flere lag
# vi kan ikke optimere uden gradients
# prøv med forskellige hyperparameter
# gaussian search for hyperparameters can help. Libraries
# Hyperparameter search.

# spørgsmål performance:
#


class ConvLinearNN2(Model):
    def __init__(self, x_size, y_size, action_space, dim=3):
        super(ConvLinearNN2, self).__init__()
        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys

        ks = 8
        self.conv1 = nn.Conv2d(1, 8, ks, bias=True)
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 5
        self.conv2 = nn.Conv2d(8, 16, ks, bias=True)
        img_x -= (ks - 1)
        img_y -= (ks - 1)


        ks = 5
        ks_stride = 2
        self.max_pool2d = nn.MaxPool2d(ks, stride=ks_stride)

        img_x -= (ks - 1)
        img_x /= ks_stride
        img_y -= (ks - 1)
        img_y /= ks_stride

        img_x -= (ks - 1)
        img_x /= ks_stride
        img_y -= (ks - 1)
        img_y /= ks_stride

        img_x = int(img_x)
        img_y = int(img_y)

        img_size = (16 * img_x * img_y)
        print(img_size)
        self.dense1 = nn.Linear(img_size, 100)
        self.dense2 = nn.Linear(100, action_space)

    def forward(self, x):
        #x = torch.FloatTensor(x)
        #x = torch.tensor(x).float()
        x = x.to(self.device)

        #x = T.resize(x, size=(self.xs, self.ys))
        #x = T.rgb_to_grayscale(x)

        m = self.max_pool2d
        #batch_norm = nn.BatchNorm2d()

        x = self.conv1(x)  # convolutional
        x = F.relu(x)
        x = m(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = m(x)

        x = torch.flatten(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        #print(x)
        return torch.argmax(x)

class ConvLinearNNMult(Model):
    def __init__(self, x_size, y_size, action_space, batch_size=1024):
        super(ConvLinearNNMult, self).__init__()
        self.batch_size = batch_size

        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys

        ks = 3
        #self.conv1 = nn.Conv2d(1, 8, ks, bias=True)
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, ks, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        #self.conv2 = nn.Conv2d(8, 16, ks, bias=True)
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, ks, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)


        ks = 3
        ks_stride = 1

        self.img_size = (64 * img_x * img_y)
        print(self.img_size)

        self.dense1 = nn.Sequential(
            nn.Linear(self.img_size, 100),
            nn.ReLU()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(100, action_space)
        )

    def forward(self, x):
        x = x.to(self.device)

        x = self.conv1(x)

        x = self.conv2(x)

        x = x.view(-1, self.img_size)

        x = self.dense1(x)
        x = self.dense2(x)

        return x

    def predict(self, x):
        x = self.forward(x)
        #print(x)
        return torch.argmax(x)

class DuelNetwork(Model):
    def __init__(self, x_size, y_size, action_space, batch_size=1024):
        super(DuelNetwork, self).__init__()
        self.batch_size = batch_size

        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys

        ks = 3
        #self.conv1 = nn.Conv2d(1, 8, ks, bias=True)
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, ks,  bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        #self.conv2 = nn.Conv2d(8, 16, ks, bias=True)
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, ks,  bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        
        ks = 3
        self.conv3 = nn.Sequential(
                nn.Conv2d(32, 32, ks,  bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)
        
        
        ks = 3
        self.conv4 = nn.Sequential(
                nn.Conv2d(32, 64, ks,  bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        ks_stride = 1
        
        self.img_size = (64 * img_x * img_y)
        print(self.img_size)

        linearInputSize = int(self.img_size/2)

        self.state_value = nn.Sequential(
            nn.Linear(linearInputSize, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.advantage_value = nn.Sequential(
            nn.Linear(linearInputSize, 100),
            nn.ReLU(),
            nn.Linear(100, action_space)
        )

    def forward(self, x):
        x = x.to(self.device)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = x.view(-1, self.img_size)

        sliceSize = int(self.img_size/2)

        x1 = x[:, :sliceSize]
        x2 = x[:, sliceSize:]

        state_value = self.state_value(x1).reshape(-1,1)
        advantage_value = self.advantage_value(x2)

        q = state_value + (advantage_value - advantage_value.mean(dim=1).reshape(-1, 1))

        return q

    def predict(self, x):
        x = self.forward(x)
        #print(x)
        return torch.argmax(x)