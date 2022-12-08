import random
from collections import deque  # for memory
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src import Models
from src.Agents.Agent import AgentBase


class AgentDQN(AgentBase):
    def __init__(self, memory_size=10000, model_name='default_DQN_model', learning_rate=1e-4, batch_size=64,
                 frame_stack_size=1):
        super().__init__(learning_rate=learning_rate, model_name=model_name)
        self.N = memory_size
        self.memory = None
        self.norm_rewards = None
        self.batch_size = batch_size
        self.frame_stack_size = frame_stack_size

    def get_action(self, state, explore=True):
        if random.random() < self.exploration and explore:
            action_index = random.randint(0, len(self.actions)-1)
            action = self.actions[action_index]
        else:
            state = np.expand_dims(state, axis=0)
            with torch.no_grad():
                state = torch.from_numpy(state).float().cpu()
                action_index = int(self.model.predict(state))
            action = self.actions[action_index]

        return action

    def remember(self, state, action, reward, next_state, done):
        action = self.actions.index(action)
        self.memory.append([state, action, reward, next_state, done])

    def train(self, state, action, next_state, reward, done=False):
        self.remember(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            loss = self.replay(self.batch_size)
            return loss

        return 0

    def replay(self, batch_size):
        #self.normalize_rewards()
        minibatch = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()

        minibatch = np.array(minibatch.copy(), dtype=object)

        states = torch.from_numpy(np.stack(minibatch[:, 0]).astype(float)).float().to(self.device)
        actions = torch.from_numpy(np.array(minibatch[:, 1]).astype(np.int64)).long().to(self.device)
        rewards = torch.from_numpy(np.array(minibatch[:, 2]).astype(float)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(minibatch[:, 3]).astype(float)).float().to(self.device)
        dones = torch.from_numpy(np.array(minibatch[:, 4]).astype(bool)).to(self.device)
        not_dones = ~dones
        not_dones = not_dones.int()

        row = np.arange(self.batch_size)

        with torch.no_grad():
            nsi = row, torch.argmax(self.model.forward(next_states), dim=1)  # nsi = next state indices
            next_state_values = self.model.forward(next_states)[nsi] #

        v = rewards + self.dr * next_state_values * not_dones

        a = row, actions
        p = self.model.forward(states)[a]
        loss = self.criterion(p, v)

        if float(loss.item()) > 10_000:
            print("wowsers")

        loss.backward()

        self.optimizer.step()
        return loss.item()

    def load_model_config(self, tune_config):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.lr = tune_config["lr"]

        self.criterion = nn.MSELoss()
        #self.criterion = nn.L1Loss()
        self.model = Models.DuelNetworkConfigurable(self.downscale[0], self.downscale[1],
                                                    len(self.actions),  self.frame_stack_size,
                                                    c1=tune_config["c1"], c2=tune_config["c2"],
                                                    c3=tune_config["c3"], c4=tune_config["c4"])

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)

        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=tune_config["lr"], momentum=tune_config["momentum"])
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def set_model(self, model):
        self.model = model

    def get_model(self):
        if self.model is None:
            return Models.DQNModel(self.downscale[0], self.downscale[1], len(self.actions), stack_size=self.frame_stack_size)
        else:
            return self.model(self.downscale[0], self.downscale[1], len(self.actions), stack_size=self.frame_stack_size)

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.criterion = nn.MSELoss()
        #self.criterion = nn.L1Loss()

        self.model = self.get_model()

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)

        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def start_training(self, config, epoch_count=10, episodes_per_epoch=100, episodes_per_test=10, tics_per_action=12, hardcoded_path=False,
                       fast_train=True, tune_config=None):
        if tics_per_action < self.frame_stack_size:
            print("tics per action can not be less than frames per step")
            return

        # Set up game environment and action
        self.set_up_game_environment(config, hardcoded_path)
        game = self.game
        if tune_config == None:
            self.load_model()
        else:
            self.load_model_config(tune_config)

        # Set up ray and training details
        writer = SummaryWriter(comment=('_'+self.model_name))
        writer.filename_suffix = self.model_name
        first_run = False

        # Epoch runs a certain amount of episodes, followed a test run to show performance.
        # At the end the model is saved on disk
        for epoch in range(epoch_count):
            print("epoch: ", epoch+1)

            for e in trange(episodes_per_epoch):
                if fast_train:
                    loss = self.train_run_fast(tics_per_action, first_run)
                else:
                    loss = self.train_run(tics_per_action, first_run)

                writer.add_scalar('Loss_epoch_size_' + str(episodes_per_epoch), loss, e + epoch * episodes_per_epoch)
                writer.add_scalar('Reward_epoch_size_' + str(episodes_per_epoch), game.get_total_reward(),
                                  e + epoch * episodes_per_epoch)
                writer.add_scalar('Exploration_epoch_size_' + str(episodes_per_epoch), self.exploration,
                                  e + epoch * episodes_per_epoch)

            self.save_model()
            avg_score = 0.0
            for e in trange(episodes_per_test):
                avg_score += self.test_run_fast(tics_per_action)

            avg_score /= episodes_per_test
            writer.add_scalar('Score_epoch_size_' + str(episodes_per_test), avg_score, epoch)

            first_run = False


class AgentDuelDQN(AgentDQN):
    def __init__(self, memory_size=10000, model_name='default_DuelDQN_model', learning_rate=1e-4, batch_size=64):
        super().__init__(memory_size=memory_size, learning_rate=learning_rate, model_name=model_name, batch_size=batch_size)

    def get_model(self):
        if self.model is None:
            return Models.DuelNetworkConfigurable(self.downscale[0], self.downscale[1], len(self.actions), self.frame_stack_size)
        else:
            return self.model(self.downscale[0], self.downscale[1], len(self.actions), self.frame_stack_size)


class AgentDoubleDuelDQN(AgentDuelDQN):
    def __init__(self, memory_size=10000, model_name='default_DoubleDuelDQN_model', learning_rate=1e-4, batch_size=64):
        super().__init__(memory_size=memory_size, learning_rate=learning_rate, model_name=model_name, batch_size=batch_size)

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        #self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()

        self.model = self.get_model()
        self.target = self.get_model()

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.target.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)
        self.target.set_device(self.device)
        self.target.to(self.device)

        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()

        minibatch = np.array(minibatch.copy(), dtype=object)

        states = torch.from_numpy(np.stack(minibatch[:, 0]).astype(float)).float().to(self.device)
        actions = torch.from_numpy(np.array(minibatch[:, 1]).astype(np.int64)).long().to(self.device)
        rewards = torch.from_numpy(np.array(minibatch[:, 2]).astype(float)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(minibatch[:, 3]).astype(float)).float().to(self.device)
        dones = torch.from_numpy(np.array(minibatch[:, 4]).astype(bool)).to(self.device)
        not_dones = ~dones
        not_dones = not_dones.int()

        row = np.arange(self.batch_size)

        with torch.no_grad():
            nsi = row, torch.argmax(self.target.forward(next_states), dim=1)  # nsi = next state indices
            next_state_values = self.target.forward(next_states)[nsi] #

        v = rewards + self.dr * next_state_values * not_dones

        a = row, actions
        p = self.model.forward(states)[a]

        loss = self.criterion(v, p)
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        self.target.load_state_dict(self.model.state_dict())
        print("model saved")
