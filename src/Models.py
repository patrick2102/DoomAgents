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
        self.device = device

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


class ConvLinearNN2(Model):
    def __init__(self, x_size, y_size, action_space, dim=3):
        super(ConvLinearNN2, self).__init__()
        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys
        self.blur_kernel = int(1 / rescale_factor)
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

        ks = 6
        self.conv1 = nn.Conv2d(1, 8, ks, bias=False)
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        self.conv2 = nn.Conv2d(8, 8, ks, bias=False)
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        self.conv3 = nn.Conv2d(8, 1, ks, bias=False)
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        """
        self.conv4 = nn.Conv2d(8, 1, ks, bias=False)
        img_x -= (ks - 1)
        img_y -= (ks - 1)
        """

        img_size = (img_x * img_y)
        self.dense1 = nn.Linear(img_size, 100)
        self.dense2 = nn.Linear(100, action_space)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = x.to(self.device)

        x = T.resize(x, size=(self.xs, self.ys))
        x = T.rgb_to_grayscale(x)

        #m = nn.BatchNorm1d(8)

        x = self.conv1(x)  # convolutional
        #x = m(x)
        x = F.relu(x)

        x = self.conv2(x)
        #x = m(x)
        x = F.relu(x)

        x = self.conv3(x)
        #x = m(x)
        #x = F.relu(x)

        #m = nn.BatchNorm1d(16)

        #x = self.conv4(x)
        #x = m(x)
        #x = F.relu(x)

        x = torch.flatten(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        print(x)
        return torch.argmax(x)
