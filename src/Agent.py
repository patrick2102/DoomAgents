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
    def __init__(self, memory_size=30000, model_name='DuelDQNInitial3'):
        super().__init__()
        self.criterion = None
        self.model = None
        self.optimizer = None
        self.N = memory_size
        self.memory = None
        self.batch_size = 64
        self.exploration = 1.0
        self.exploration_decay = 0.9995
        self.min_exploration = 0.1
        self.downscale = (30, 45)
        self.model_path = 'models/'+model_name+'.pth'

    def preprocess(self, state):
        s = state.screen_buffer

        s = np.moveaxis(s, 0, 2)

        s = cv2.resize(s, self.downscale, interpolation=cv2.INTER_AREA)
        s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)

        s = np.moveaxis(s, 1, 0)

        s = np.array(s, dtype=float) / 255

        s = np.expand_dims(s, axis=0)

        return s

    def decay_exploration(self):
        if len(self.memory) >= self.batch_size:
            self.exploration *= self.exploration_decay
            if self.exploration < self.min_exploration:
                self.exploration = self.min_exploration
        print("exploration: ", self.exploration)

    def remember(self, state, action, reward, next_state, done):
        state = self.preprocess(state)
        if next_state is None:
            next_state = np.zeros((1, 30, 45)).astype(np.float32)
        else:
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

    def update_q(self, sample):
        s, a, r, s1, d = sample

        r = torch.tensor(r)
        if not d:
            v = r + 0.99 * torch.max(self.model.forward(s1))
        else:
            v = r

        v = v.to(self.device)

        pred = self.model.forward(s)[a]
        loss = self.criterion(pred, v)
        loss.backward()

        return int(loss)

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.criterion = nn.MSELoss()
        self.model = Models.DuelNetwork(self.downscale[0], self.downscale[1], len(self.actions))

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)

        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4)
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("model saved")

class AgentDoubleDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.criterion = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.actor = None
        self.critic = None
        self.last_state = None
        self.last_action = 0
        self.device = "cpu"
        #self.batch_size = 64
        self.exploration = 0.0
        self.exploration_decay = 1.0
        self.train_actor = True

    def init_model(self, criterion, model: src.Models.Model, lr=1e-5):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.criterion = criterion

        self.actor = copy.deepcopy(model)
        self.critic = copy.deepcopy(model)

        self.actor.set_device(self.device)
        self.critic.set_device(self.device)

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.criterion.to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def decay_exploration(self):
        self.exploration *= self.exploration_decay
        print("exploration: ", self.exploration)

    def get_image(self, state):
        s = state.screen_buffer
        s = np.array(s, dtype=float)/255
        return s

    def show_image(self, img):
        cv2.imshow("t", img)

    def get_action(self, s):
        state = self.get_image(s)

        if random.random() < self.exploration:
            action_index = random.randint(0, len(self.actions)-1)
            action = self.actions[action_index]
        else:
            if self.train_actor:
                action_index = int(self.actor.predict(state))
            else:
                action_index = int(self.critic.predict(state))
            action = self.actions[action_index]

        self.last_state = state
        self.last_action = action_index
        return action

    def train(self, next_state: GameState, reward, done=False):
        s = self.last_state
        a = self.last_action

        r = torch.tensor(reward)

        if not done:
            s1 = self.get_image(next_state)

            #v = r + 0.99 * float(torch.max(self.model.forward(s1)))
            #v_actor = r + 0.99 * self.critic.forward(s1)[torch.argmax(self.actor.forward(s1))]
            #v_critic = r + 0.99 * self.actor.forward(s1)[torch.argmax(self.critic.forward(s1))]

            if self.train_actor:
                v = r + 0.99 * torch.max(self.critic.forward(s1))
            else:
                v = r + 0.99 * torch.max(self.actor.forward(s1))
        else:
            v = r

        v = v.to(self.device)

        if self.train_actor:
            pred = self.actor.forward(s)[a]
        else:
            pred = self.critic.forward(s)[a]

        loss = self.criterion(pred, v)
        loss.backward()

        if done:
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

        self.train_actor = not self.train_actor

        return int(loss)

    def save_model(self):
        torch.save(self.actor.state_dict(), 'models/DDQN_actor4.pth')
        torch.save(self.critic.state_dict(), 'models/DDQN_critic4.pth')
        print("model saved")

    def load_model(self):
        self.actor.load_state_dict(torch.load('models/DDQN_actor4.pth'))
        self.critic.load_state_dict(torch.load('models/DDQN_critic4.pth'))
        print("model loaded")