import random
import numpy as np
import skimage
import torch
import torch.nn as nn
import torch.optim as optim
import itertools as it
import torch.nn.functional as F
import cv2 as cv2
from vizdoom import *
from collections import deque  # for memory
import copy
import src.Models
from src import Models
from os.path import exists
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from skimage import io
from skimage.viewer import ImageViewer
from sklearn import preprocessing



class AgentBase:
    def __init__(self, learning_rate=1e-4, model_name='default_model'):
        self.actions = []
        self.model_name = model_name
        self.model_path = 'models/'+model_name+'.pth'
        self.lr = learning_rate
        self.criterion = None
        self.model = None
        self.optimizer = None
        self.exploration = 0.1
        self.exploration_decay = 0.9995
        self.min_exploration = 0.1
        self.downscale = (30, 45)
        self.frame_stack_size = 4
        self.game = None

    def set_model_path(self, model_name):
        self.model_path = 'models/'+model_name+'.pth'

    def preprocess(self, state):
        s = state

        s = cv2.resize(s, self.downscale, interpolation=cv2.INTER_AREA)

        s = np.moveaxis(s, 1, 0)

        s = np.array(s, dtype=float)/255

        return s

    def set_up_game_environment(self, config, hardcoded_path):
        # Set up game environment
        self.game = DoomGame()
        if hardcoded_path:
            config_path = "C:/Uni/3rd_Semester/DeepLearning/Project/DoomAgents/" + config
        else:
            config_path = config
        self.game.load_config(config_path)
        self.game.init()

        # Set up model and possible actions
        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]

    def set_available_actions(self, avail_actions):
        self.actions = avail_actions

    def get_action(self, state, explore=True):
        raise NotImplementedError

    def train(self, state, last_action, next_state, reward,  done=False):
        raise NotImplementedError

    """
    def start_training(self, config, epoch_count=10, episodes_per_epoch=100, episodes_per_test=10, tics_per_action=12, hardcoded_path=False,
                       fast_train=False, tune_config=None):
        raise NotImplementedError
    """
    """
        Run without exploration to measure performance
    """
    def test_run(self, tics_per_action=12):
        game = self.game
        game.new_episode()
        done = False
        prev_frames = deque([np.zeros(self.downscale).astype(np.float32)] * self.frame_stack_size,
                            maxlen=self.frame_stack_size)

        while not done:
            frame = self.preprocess(game.get_state().screen_buffer)
            prev_frames.append(frame)
            state = np.array(prev_frames)
            action = self.get_action(state, explore=False)
            game.set_action(action)
            for i in range(tics_per_action):
                game.advance_action()
                done = game.is_episode_finished()
                if done:
                    frame = np.zeros(self.downscale).astype(np.float32)
                    prev_frames.append(frame)
                    break
                else:
                    frame = self.preprocess(game.get_state().screen_buffer)
                    prev_frames.append(frame)

        return game.get_total_reward()

    """
        Train model
    """
    def train_run(self, tics_per_action, first_run):
        game = self.game
        game.new_episode()
        prev_frames = deque([np.zeros(self.downscale).astype(np.float32)] * self.frame_stack_size,
                            maxlen=self.frame_stack_size)
        done = False
        loss = 0
        steps = 0

        while not done:
            steps += 1
            frame = self.preprocess(game.get_state().screen_buffer)
            prev_frames.append(frame)

            state = np.array(prev_frames)
            action = self.get_action(state)
            game.set_action(action)
            reward = 0

            for i in range(tics_per_action):
                game.advance_action()
                reward += game.get_last_reward()
                done = game.is_episode_finished()
                if done:
                    frame = np.zeros(self.downscale).astype(np.float32)
                    prev_frames.append(frame)
                    break
                else:
                    frame = self.preprocess(game.get_state().screen_buffer)
                    prev_frames.append(frame)

            next_state = np.array(prev_frames)

            loss += self.train(state, action, next_state, reward, done)

            if not first_run:
                self.decay_exploration()

        loss /= steps

        return loss

    def load_model_config(self, tune_config):
        raise NotImplementedError

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("model saved")

    def decay_exploration(self):
        if len(self.memory) >= self.batch_size:
            self.exploration *= self.exploration_decay
            if self.exploration < self.min_exploration:
                self.exploration = self.min_exploration

class AgentRandom(AgentBase):
    def get_action(self, state: GameState):
        return random.choice(self.actions)

class AgentDQN(AgentBase):
    def __init__(self, memory_size=10000, model_name='default_DQN_model', learning_rate=1e-4, batch_size=64,
                 frame_stack_size=4):
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
        reward /= 10.0
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

        states = torch.from_numpy(np.stack(minibatch[:, 0]).astype(np.double)).float().to(self.device)
        actions = torch.from_numpy(np.array(minibatch[:, 1]).astype(np.int64)).long().to(self.device)
        rewards = torch.from_numpy(np.array(minibatch[:, 2]).astype(float)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(minibatch[:, 3]).astype(np.double)).float().to(self.device)
        dones = torch.from_numpy(np.array(minibatch[:, 4]).astype(bool)).to(self.device)
        not_dones = ~dones
        not_dones = not_dones.int()

        row = np.arange(self.batch_size)

        with torch.no_grad():
            nsi = row, torch.argmax(self.model.forward(next_states), dim=1)  # nsi = next state indices
            next_state_values = self.model.forward(next_states)[nsi] #

        v = rewards + 0.99 * next_state_values * not_dones

        a = row, actions
        p = self.model.forward(states)[a]

        loss = self.criterion(v, p)
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def load_model_config(self, tune_config):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.lr = tune_config["lr"]

        self.criterion = nn.MSELoss()
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

    def get_model(self):
        return Models.DQNModel(self.downscale[0], self.downscale[1], len(self.actions), stack_size=self.frame_stack_size)

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.criterion = nn.MSELoss()

        self.model = self.get_model()

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)

        self.criterion.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.memory = deque([], maxlen=self.N)
        print("model loaded")


    """
        Train run fast does not support multiple images
    """
    def train_run_fast(self, tics_per_action, first_run):
        game = self.game
        # Training loop where learning happens
        game.new_episode()
        done = False
        loss = 0
        steps = 0

        while not done:
            steps += 1
            frame = self.preprocess(game.get_state().screen_buffer)
            # Quick and dirty solution that makes train_run_fast work without replacing the model.
            # Might ruin training if combined with train_run
            state = np.array([frame] * self.frame_stack_size).astype(np.float32)

            action = self.get_action(state)
            reward = game.make_action(action, tics_per_action)

            done = game.is_episode_finished()

            if not done:
                frame = self.preprocess(game.get_state().screen_buffer)
            else:
                frame = np.zeros(self.downscale).astype(np.float32)

            next_state = np.array([frame] * self.frame_stack_size).astype(np.float32)

            loss += self.train(state, action, next_state, reward, done)

            if not first_run:
                self.decay_exploration()

        loss /= steps

        return loss




    """
        Run without multiple images and exploration
    """
    def test_run_fast(self, tics_per_action):
        game = self.game
        game.new_episode()
        done = False
        prev_frames = deque([np.zeros(self.downscale).astype(np.float32)] * self.frame_stack_size,
                            maxlen=self.frame_stack_size)

        while not done:
            frame = self.preprocess(game.get_state().screen_buffer)
            prev_frames.append(frame)
            state = np.array(prev_frames)
            action = self.get_action(state, explore=False)
            game.make_action(action, tics_per_action)

        return game.get_total_reward()



    def start_training(self, config, epoch_count=10, episodes_per_epoch=100, episodes_per_test=10, tics_per_action=12, hardcoded_path=False,
                       fast_train=False, tune_config=None):
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

            for e in trange(episodes_per_test):
                self.test_run(tics_per_action)
                writer.add_scalar('Score_epoch_size_' + str(episodes_per_epoch), game.get_total_reward(),
                                  e + epoch * episodes_per_test)

            first_run = False

class AgentDuelDQN(AgentDQN):
    def __init__(self, memory_size=10000, model_name='default_DuelDQN_model', learning_rate=1e-4, batch_size=64):
        super().__init__(memory_size=memory_size, learning_rate=learning_rate, model_name=model_name, batch_size=batch_size)

    def get_model(self):
        return Models.DuelNetworkConfigurable(self.downscale[0], self.downscale[1], len(self.actions), self.frame_stack_size)

class AgentDoubleDuelDQN(AgentDuelDQN):
    def __init__(self, memory_size=10000, model_name='default_DoubleDuelDQN_model', learning_rate=1e-4, batch_size=64):
        super().__init__(memory_size=memory_size, learning_rate=learning_rate, model_name=model_name, batch_size=batch_size)

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

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
            nsi = row, torch.argmax(self.model.forward(next_states), dim=1)  # nsi = next state indices
            next_state_values = self.target.forward(next_states)[nsi] #

        v = rewards + 0.99 * next_state_values * not_dones

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


class A2C(AgentBase):
    def __init__(self, memory_size=10000, model_name='default_A2C_model', learning_rate=1e-4, batch_size=64):
        super().__init__(learning_rate=learning_rate, model_name=model_name)
        self.N = memory_size
        self.memory = None
        self.norm_rewards = None
        self.batch_size = 64
        self.frame_stack_size = 4

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

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()

        minibatch = np.array(minibatch.copy(), dtype=object)

        states = torch.from_numpy(np.stack(minibatch[:, 0]).astype(np.double)).float().to(self.device)
        actions = torch.from_numpy(np.array(minibatch[:, 1]).astype(np.int64)).long().to(self.device)
        rewards = torch.from_numpy(np.array(minibatch[:, 2]).astype(float)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(minibatch[:, 3]).astype(np.double)).float().to(self.device)
        dones = torch.from_numpy(np.array(minibatch[:, 4]).astype(bool)).to(self.device)
        not_dones = ~dones
        not_dones = not_dones.int()

        row = np.arange(self.batch_size)

        a = row, actions
        v, pd = self.model.forward(states)
        with torch.no_grad():
            ns_v, _ = self.model.forward(next_states)

        ns_v = np.squeeze(ns_v)
        v = np.squeeze(v)

        q = rewards + 0.99 * ns_v * not_dones

        advantage = q - v

        pd = pd.squeeze(0)

        pd = pd[a]

        pd = torch.log(pd)

        actor_loss = -pd*advantage
        critic_loss = advantage**2

        #loss = torch.sqrt((actor_loss + critic_loss)**2)
        loss = torch.abs((actor_loss + critic_loss).mean())

        #loss = self.criterion(actor_loss, critic_loss)

        loss.backward()
        #actor_loss.backward()
        #critic_loss.backward()

        self.optimizer.step()
        return loss.item()

    def remember(self, state, action, reward, next_state, done):
        action = self.actions.index(action)
        reward /= 10.0
        self.memory.append([state, action, reward, next_state, done])

    def train(self, state, action, next_state, reward, done=False):
        self.remember(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            loss = self.replay(self.batch_size)
            return loss

        return 0

    def get_model(self):
        return Models.ActorCritic(self.downscale[0], self.downscale[1], len(self.actions),
                                  stack_size=self.frame_stack_size)

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.criterion = nn.MSELoss()

        self.model = self.get_model()

        if exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        self.model.set_device(self.device)
        self.model.to(self.device)

        self.criterion.to(self.device)

        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def start_training(self, config, epoch_count=10, episodes_per_epoch=100, episodes_per_test=10, tics_per_action=12, hardcoded_path=False,
                       fast_train=False, tune_config=None):
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

            for e in trange(episodes_per_test):
                self.test_run(tics_per_action)
                writer.add_scalar('Score_epoch_size_' + str(episodes_per_epoch), game.get_total_reward(),
                                  e + epoch * episodes_per_test)

            first_run = False


class A2CPPO(A2C):
    def __init__(self, memory_size=10000, model_name='default_A2C_model', learning_rate=1e-4, batch_size=64):
        super().__init__(learning_rate=learning_rate, model_name=model_name)
        self.old_pd = None
        self.max_al = 0.0
        self.max_cl = 0.0

    def remember(self, state, action, reward, next_state, done):
        action = self.actions.index(action)
        reward /= 10.0
        with torch.no_grad():
            s = copy.copy(state)
            s = np.expand_dims(s, axis=0)
            s = torch.from_numpy(s).float().cpu()
            v, old_pd = self.model.forward(s)
            old_pd = np.squeeze(old_pd)
            old_pd = old_pd[action]
            #old_pd = float(old_pd.cpu().numpy())
            old_pd = old_pd.cpu().numpy()

            #advantage =
        # state, action, reward, next state, done, old_policy
        self.memory.append([state, action, reward, next_state, done, old_pd])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()

        minibatch_og = minibatch
        minibatch = np.array(minibatch, dtype=object)

        states = torch.from_numpy(np.stack(minibatch[:, 0]).astype(np.double)).float().to(self.device)
        actions = torch.from_numpy(np.array(minibatch[:, 1]).astype(np.int64)).long().to(self.device)
        rewards = torch.from_numpy(np.array(minibatch[:, 2]).astype(float)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(minibatch[:, 3]).astype(np.double)).float().to(self.device)
        dones = torch.from_numpy(np.array(minibatch[:, 4]).astype(bool)).to(self.device)
        old_pds = torch.from_numpy(np.array(minibatch[:, 5]).astype(float)).float().to(self.device)
        not_dones = ~dones
        not_dones = not_dones.int()

        row = np.arange(self.batch_size)

        a = row, actions
        v, pd = self.model.forward(states)
        with torch.no_grad():
            ns_v, _ = self.model.forward(next_states)

        ns_v = np.squeeze(ns_v)
        v = np.squeeze(v)

        q = rewards + 0.99 * ns_v * not_dones

        advantage = q - v

        pd_logs = torch.log(pd)
        with torch.no_grad():
            entropy = pd * pd_logs
            entropy = -torch.sum(entropy, dim=1)
            pd = pd.squeeze(0)



        #loss = torch.abs((actor_loss + critic_loss).mean())

        c1 = 1.0
        c2 = 0.01

        old_pd_logs = torch.log(old_pds)

        #ratios = torch.exp((pd_logs[a]+1e-10)-(old_pd_logs+1e-10))
        ratios = pd[a]/old_pd_logs

        eps = 0.2

        s1 = ratios * advantage

        # https://spinningup.openai.com/en/latest/algorithms/ppo.html

        b = torch.clamp(ratios, 1-eps, 1+eps) * advantage
        s2 = b

        #b = torch.clamp(advantage, 0, 1)
        #torch.ceil(b)
        #b = b - (1-b)
        #s2 = (1+(eps*b))*advantage

        actor_loss = -torch.mean(torch.min(s1, s2))

        critic_loss = torch.mean((rewards-v)**2)

        if self.max_cl < float(critic_loss):
            self.max_cl = float(critic_loss)
            print("max critic loss: ", float(critic_loss))

        if self.max_al < float(actor_loss):
            self.max_al = float(actor_loss)
            print("s1: ", s1)
            print("s2: ", s2)
            print("max actor  loss: ", float(actor_loss))

        loss = c1 * (actor_loss + critic_loss) #+ c2 * entropy

        #loss = surr + c1*(ns_v - v)**2 + c2 * (-torch.sum(pd*pd_log))
        #loss = c1 * (actor_loss + critic_loss)
        #loss += c2 * entropy
        #loss = surr + c1*(actor_loss+critic_loss) + c2 * (-torch.sum(pd*pd_log))
        #loss = torch.abs(loss.mean())
        #loss = loss.mean()
        #loss = torch.abs(loss.mean())

        loss.backward()

        self.optimizer.step()

        #old_pds = pd.cpu().detach().numpy()

        #old_pds = pd

        #minibatch[:, 5] = pd1
        #minibatch_og[:, 5] = pd1

        # Temp solution, should be calculated using vectors instead
        pd = pd[a]

        for i in range(self.batch_size):
            minibatch_og[i][5] = float(pd[i])

        return loss.item()
