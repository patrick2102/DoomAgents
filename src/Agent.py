import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2 as cv2
from vizdoom import *
from collections import deque  # for memory
import copy
import src.Models
from src import Models
from os.path import exists


class AgentBase:
    def __init__(self):
        self.actions = []

    def set_available_actions(self, avail_actions):
        self.actions = avail_actions

    def get_action(self, state: GameState):
        raise NotImplementedError

    def train(self, last_state, last_action, state: GameState, reward,  done=False):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError


class AgentRandom(AgentBase):
    def get_action(self, state: GameState):
        return random.choice(self.actions)


class AgentDQN(AgentBase):
    def __init__(self, memory_size=20000, model_name='DQNInitial'):
        super().__init__()
        self.criterion = None
        self.model = None
        self.optimizer = None
        self.N = memory_size
        self.memory = None
        self.lr = 0.0001
        self.batch_size = 64
        self.exploration = 0.1
        self.exploration_decay = 0.999
        self.min_exploration = 0.1
        self.downscale = (30, 45)
        self.model_path = 'models/'+model_name+'.pth'
        self.stack_size = 4
        self.state_stack = deque([], maxlen=self.stack_size)
        self.next_state_stack = deque([], maxlen=self.stack_size)

    def preprocess(self, state):
        states = []
        for s in state:
            if s is None:
                s = np.zeros(self.downscale).astype(np.float32)
            else:
                s = np.moveaxis(s, 0, 2)

                s = cv2.resize(s, self.downscale, interpolation=cv2.INTER_AREA)

                s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)

                cv2.imshow("image", s)

                s = np.moveaxis(s, 1, 0)

                s = np.array(s, dtype=float) / 255

                #s = np.expand_dims(s, axis=0)

            states.append(s)

        states = np.array(states)
        #states = np.moveaxis(states, 0, 1)

        return states

    def decay_exploration(self):
        if len(self.memory) >= self.batch_size:
            self.exploration *= self.exploration_decay
            if self.exploration < self.min_exploration:
                self.exploration = self.min_exploration
        print("exploration: ", self.exploration)

    def remember(self, state, action, reward, next_state, done):
        state = self.preprocess(state)
        next_state = self.preprocess(next_state)

        action = self.actions.index(action)
        self.memory.append([state, action, reward, next_state, done])

    def get_action(self, state):

        if random.random() < self.exploration:
            action_index = random.randint(0, len(self.actions)-1)
            action = self.actions[action_index]
        else:
            state = self.preprocess(state)
            state = [torch.tensor(state).float().cpu()]
            state = torch.stack(state)
            with torch.no_grad():
                action_index = int(self.model.predict(state))
            action = self.actions[action_index]

        return action

    def train(self, state, action, next_state: GameState, reward, done=False):

        self.remember(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            avg_loss = self.replay(self.batch_size)
            return avg_loss

        """
        if done:
            if len(self.memory) >= self.batch_size:
                avg_loss = self.replay(self.batch_size)
                return avg_loss
        """

        return 0

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()

        total_loss = 0

        minibatch = np.array(minibatch.copy(), dtype=object)

        states = torch.from_numpy(np.stack(minibatch[:, 0]).astype(float)).float().to(self.device)
        actions = torch.from_numpy(np.array(minibatch[:, 1]).astype(int)).int().to(self.device)
        rewards = torch.from_numpy(np.array(minibatch[:, 2]).astype(float)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(minibatch[:, 3]).astype(float)).float().to(self.device)
        dones = torch.from_numpy(np.array(minibatch[:, 4]).astype(bool)).to(self.device)
        not_dones = ~dones
        not_dones = not_dones.int()

        v = rewards + 0.99 * torch.max(self.model.forward(next_states), dim=1)[0] * not_dones

        s = self.model.forward(states)
        p = []

        for i in range(batch_size):
            a = actions[i]
            p.append(s[i][a])

        p = torch.stack(p)

        loss = self.criterion(p, v)
        loss.backward()

        avg_loss = total_loss/batch_size

        self.optimizer.step()
        return avg_loss

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.criterion = nn.MSELoss()
        self.model = Models.ConvLinearNNMult(self.downscale[0], self.downscale[1],
                                             len(self.actions), self.stack_size+1, self.batch_size)

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)

        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("model saved")

class AgentDuelDQN(AgentBase):
    def __init__(self, memory_size=10000, model_name='DuelDQNInitial'):
        super().__init__()
        self.lr = 0.0001
        self.criterion = None
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.N = memory_size
        self.memory = None
        self.batch_size = 64
        self.exploration = 1.0
        self.exploration_decay = 0.99
        self.min_exploration = 0.1
        self.downscale = (30, 45)
        self.model_path = 'models/'+model_name+'.pth'
        self.stack_size = 4
        self.state_stack = deque([], maxlen=self.stack_size)
        self.next_state_stack = deque([], maxlen=self.stack_size)

    def preprocess(self, state):
        states = []
        for s in state:
            if s is None:
                s = np.zeros(self.downscale).astype(np.float32)
            else:
                s = np.moveaxis(s, 0, 2)
                #cv2.imshow("image", s)

                s = cv2.resize(s, self.downscale, interpolation=cv2.INTER_AREA)
                s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)


                s = np.moveaxis(s, 1, 0)

                s = np.array(s, dtype=float) / 255

                #s = np.expand_dims(s, axis=0)

            states.append(s)

        states = np.array(states)
        #states = np.moveaxis(states, 0, 1)

        return states

    def decay_exploration(self):
        if len(self.memory) >= self.batch_size:
            self.exploration *= self.exploration_decay
            if self.exploration < self.min_exploration:
                self.exploration = self.min_exploration
        print("exploration: ", self.exploration)

    def remember(self, state, action, reward, next_state, done):
        state = self.preprocess(state)
        next_state = self.preprocess(next_state)

        action = self.actions.index(action)
        self.memory.append([state, action, reward, next_state, done])

    def get_action(self, state, explore=True):

        if random.random() < self.exploration and explore:
            action_index = random.randint(0, len(self.actions)-1)
            action = self.actions[action_index]
        else:
            state = self.preprocess(state)
            state = [torch.tensor(state).float().cpu()]
            state = torch.stack(state)
            with torch.no_grad():
                action_index = int(self.model.predict(state))
            action = self.actions[action_index]

        return action

    def train(self, state, action, next_state: GameState, reward, done=False):

        self.remember(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            loss = self.replay(self.batch_size)
            return loss

        """
        if done:
            if len(self.memory) >= self.batch_size:
                avg_loss = self.replay(self.batch_size)
                return avg_loss
        """

        return 0

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()

        total_loss = 0

        minibatch = np.array(minibatch.copy(), dtype=object)

        states = torch.from_numpy(np.stack(minibatch[:, 0]).astype(float)).float().to(self.device)
        actions = torch.from_numpy(np.array(minibatch[:, 1]).astype(int)).int().to(self.device)
        rewards = torch.from_numpy(np.array(minibatch[:, 2]).astype(float)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(minibatch[:, 3]).astype(float)).float().to(self.device)
        dones = torch.from_numpy(np.array(minibatch[:, 4]).astype(bool)).to(self.device)
        not_dones = ~dones
        not_dones = not_dones.int()

        with torch.no_grad():
            #next_state_values = torch.max(self.target_model.forward(next_states), dim=1)[0]
            next_state_values = torch.max(self.model.forward(next_states), dim=1)[0]

        v = rewards + 0.99 * next_state_values * not_dones

        s = self.model.forward(states)
        p = []

        for i in range(batch_size):
            a = actions[i]
            p.append(s[i][a])

        p = torch.stack(p)

        loss = self.criterion(p, v)
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def set_model_configuration(self, config):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.lr = config["lr"]

        self.criterion = nn.MSELoss()
        self.model = Models.DuelNetworkConfigurable(self.downscale[0], self.downscale[1],
                                                    len(self.actions),  self.stack_size,
                                                    c1=config["c1"], c2=config["c2"],
                                                    c3=config["c3"], c4=config["c4"])

        self.target_model = Models.DuelNetworkConfigurable(self.downscale[0], self.downscale[1],
                                                    len(self.actions),  self.stack_size,
                                                    c1=config["c1"], c2=config["c2"],
                                                    c3=config["c3"], c4=config["c4"])

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)
        self.target_model.set_device(self.device)
        self.target_model.to(self.device)

        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=config["lr"], momentum=config["momentum"])
        self.optimizer = optim.SGD(self.target_model.parameters(), lr=config["lr"], momentum=config["momentum"])
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.criterion = nn.MSELoss()
        self.model = Models.DuelNetworkConfigurable(self.downscale[0], self.downscale[1],
                                                    len(self.actions),  self.stack_size)
        self.target_model = Models.DuelNetworkConfigurable(self.downscale[0], self.downscale[1],
                                                    len(self.actions),  self.stack_size)

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.target_model.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)
        self.target_model.set_device(self.device)
        self.target_model.to(self.device)

        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.SGD(self.target_model.parameters(), lr=self.lr)
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("model saved")