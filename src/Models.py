import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as T
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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

class DQNModel(Model):
    def __init__(self, x_size, y_size, action_space, c1=8, c2=8, c3=8, c4=16, d1=100, stack_size=1):
        super(DQNModel, self).__init__()

        print("Running DQNModel")

        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys

        ks = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(stack_size, c1, ks, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, ks, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, ks, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, ks, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        self.img_size = (c4 * img_x * img_y)
        linearInputSize = int(self.img_size)

        self.dense = nn.Sequential(
            nn.Linear(linearInputSize, d1),
            nn.ReLU(),
            nn.Linear(d1, action_space)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.img_size)
        x = self.dense(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return torch.argmax(x)

class ConvLinearNNMult(Model):
    def __init__(self, x_size, y_size, action_space, stack_size):
        super(ConvLinearNNMult, self).__init__()

        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys

        ks = 3
        #self.conv1 = nn.Conv2d(1, 8, ks, bias=True)
        self.conv1 = nn.Sequential(
                nn.Conv2d(stack_size, 32, ks, bias=False),
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
    def __init__(self, x_size, y_size, action_space, stack_size):
        super(DuelNetwork, self).__init__()

        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys

        ks = 3
        #self.conv1 = nn.Conv2d(1, 8, ks, bias=True)
        self.conv1 = nn.Sequential(
                nn.Conv2d(stack_size, 8, ks,  bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        #self.conv2 = nn.Conv2d(8, 16, ks, bias=True)
        self.conv2 = nn.Sequential(
                nn.Conv2d(8, 8, ks,  bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        
        ks = 3
        self.conv3 = nn.Sequential(
                nn.Conv2d(8, 8, ks,  bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)
        
        
        ks = 3
        self.conv4 = nn.Sequential(
                nn.Conv2d(8, 16, ks,  bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        ks_stride = 1
        
        self.img_size = (16 * img_x * img_y)
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


class DuelNetworkConfigurable(Model):
    def __init__(self, x_size, y_size, action_space, stack_size, c1=16, c2=32, c3=32, c4=64, p=0.2):
        super(DuelNetworkConfigurable, self).__init__()

        print("Running DuelNetworkConfigurable")

        self.p = p
        self.drop_out = nn.Dropout(p)
        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys

        ks = 3
        # self.conv1 = nn.Conv2d(1, 8, ks, bias=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(stack_size, c1, ks, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        # self.conv2 = nn.Conv2d(8, 16, ks, bias=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, ks, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, ks, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, ks, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        ks_stride = 1

        self.img_size = (c4 * img_x * img_y)
        print(self.img_size)

        linearInputSize = int(self.img_size / 2)

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
        x = self.drop_out(x)

        x = self.conv2(x)
        x = self.drop_out(x)

        x = self.conv3(x)
        x = self.drop_out(x)

        x = self.conv4(x)
        x = self.drop_out(x)

        x = x.view(-1, self.img_size)

        sliceSize = int(self.img_size / 2)

        x1 = x[:, :sliceSize]
        x2 = x[:, sliceSize:]

        state_value = self.state_value(x1).reshape(-1, 1)
        advantage_value = self.advantage_value(x2)

        q = state_value + (advantage_value - advantage_value.mean(dim=1).reshape(-1, 1))

        return q

    def predict(self, x):
        x = self.forward(x)
        # print(x)
        return torch.argmax(x)

class ActorCritic(Model):
    def __init__(self, x_size, y_size, action_space, stack_size, c1=16, c2=32, c3=32, c4=64, p=0.2):
        super(ActorCritic, self).__init__()
        print("Running A2C")

        self.p = p
        self.drop_out = nn.Dropout(p)
        rescale_factor = 1.0
        self.xs = int(x_size * rescale_factor)
        self.ys = int(y_size * rescale_factor)
        img_x = self.xs
        img_y = self.ys

        ks = 3
        # self.conv1 = nn.Conv2d(1, 8, ks, bias=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(stack_size, c1, ks, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        # self.conv2 = nn.Conv2d(8, 16, ks, bias=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, ks, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, ks, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, ks, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU()
        )
        img_x -= (ks - 1)
        img_y -= (ks - 1)

        ks = 3
        ks_stride = 1

        self.img_size = (c4 * img_x * img_y)
        print(self.img_size)

        linearInputSize = int(self.img_size)

        self.critic = nn.Sequential(
            nn.Linear(linearInputSize, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(linearInputSize, 100),
            nn.ReLU(),
            nn.Linear(100, action_space)
        )

    def forward(self, x):
        x = x.to(self.device)

        x = self.conv1(x)
        #x = self.drop_out(x)

        x = self.conv2(x)
        #x = self.drop_out(x)

        x = self.conv3(x)
        #x = self.drop_out(x)

        x = self.conv4(x)
        #x = self.drop_out(x)

        x = x.view(-1, self.img_size)

        value = self.critic(x)
        policy_dist = self.actor(x)
        policy_dist = F.softmax(policy_dist, dim=1)

        return value, policy_dist

    def predict(self, x):
        _, x = self.forward(x)
        # print(x)
        return torch.argmax(x)
