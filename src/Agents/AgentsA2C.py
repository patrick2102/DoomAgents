import copy
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


class A2C(AgentBase):
    def __init__(self, memory_size=10000, model_name='default_A2C_model', learning_rate=1e-4, batch_size=64):
        super().__init__(learning_rate=learning_rate, model_name=model_name)
        self.N = memory_size
        self.memory = None
        self.exploration = 1.0
        self.min_exploration = 0.1
        self.dr = 0.99
        self.norm_rewards = None
        self.batch_size = batch_size
        self.frame_stack_size = 1
        self.actor_model = None
        self.critic_model = None
        self.actor_model_path = 'models/'+model_name+'_actor'+'.pth'
        self.critic_model_path = 'models/'+model_name+'_critic'+'.pth'

    def get_action(self, state, explore=True):
        if random.random() < self.exploration and explore:
            action_index = random.randint(0, len(self.actions)-1)
            action = self.actions[action_index]
        else:
            state = np.expand_dims(state, axis=0)
            with torch.no_grad():
                state = torch.from_numpy(state).float().cpu()
                action_index = int(self.actor_model.predict(state))
            action = self.actions[action_index]

        return action

    def train_run_fast(self, tics_per_action, first_run):
        game = self.game
        # Training loop where learning happens
        game.new_episode()
        done = False
        actor_loss = 0
        critic_loss = 0
        steps = 0

        mem = deque([], maxlen=self.N)

        while not done:
            steps += 1
            frame = self.preprocess(game.get_state().screen_buffer)
            # Quick and dirty solution that makes train_run_fast work without replacing the model.
            # Might ruin training if combined with train_run
            #state = np.array([frame]).astype(np.float32)
            state = frame

            action = self.get_action(state)
            reward = game.make_action(action, tics_per_action)

            done = game.is_episode_finished()

            if not done:
                frame = self.preprocess(game.get_state().screen_buffer)
            else:
                frame = np.zeros(self.downscale_1).astype(np.float32)

            #next_state = np.array([frame]).astype(np.float32)
            next_state = frame

            al, cl = self.train(state, action, next_state, reward, done)
            actor_loss += al
            critic_loss += cl
            #self.memory.append()
            #action = self.actions.index(action)
            #mem.append([state, action, reward, next_state, done])

            if not first_run:
                self.decay_exploration()

        actor_loss /= steps
        critic_loss /= steps

        #loss = self.replay(mem)

        return actor_loss, critic_loss

    def replay(self, batch=None):
        #minibatch = batch
        minibatch = random.sample(self.memory, self.batch_size)

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
        pd = self.actor_model.forward(states)
        v = self.critic_model.forward(states)
        with torch.no_grad():
            ns_v = self.critic_model.forward(next_states)

        ns_v = np.squeeze(ns_v)
        v = np.squeeze(v)

        q = rewards + self.dr * ns_v * not_dones

        advantage = q - v

        pd = pd.squeeze(0)[a]

        eps = 1e-5

        pd_logs = torch.log(pd+eps)

        c3 = 1e-3
        entropy = -torch.sum(pd*pd_logs)
        actor_loss = (-pd_logs*advantage)
        critic_loss = (advantage.pow(2))

        loss = torch.abs(actor_loss + critic_loss).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return actor_loss.mean().item(), critic_loss.mean().item()

    def remember(self, state, action, reward, next_state, done):
        action = self.actions.index(action)
        #reward /= 10.0
        self.memory.append([state, action, reward, next_state, done])

    def train(self, state, action, next_state, reward, done=False):
        self.remember(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            loss = self.replay(self.batch_size)
            return loss

        return 0.0, 0.0

    def save_model(self):
        torch.save(self.critic_model.state_dict(), self.critic_model_path)
        torch.save(self.actor_model.state_dict(), self.actor_model_path)
        print("model saved")

    def load_model(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

        self.criterion = nn.MSELoss()

        #self.model = self.get_model()
        self.critic_model = Models.CriticModel(self.downscale[0], self.downscale[1], len(self.actions),
                                  stack_size=self.frame_stack_size)
        self.actor_model = Models.ActorModel(self.downscale[0], self.downscale[1], len(self.actions),
                                  stack_size=self.frame_stack_size)

        if exists(self.critic_model_path):
            self.critic_model.load_state_dict(torch.load(self.critic_model_path))
            self.actor_model.load_state_dict(torch.load(self.actor_model_path))

        self.critic_model.set_device(self.device)
        self.critic_model.to(self.device)
        self.actor_model.set_device(self.device)
        self.actor_model.to(self.device)

        self.criterion.to(self.device)

        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.lr)
        self.memory = deque([], maxlen=self.N)
        print("model loaded")

    def run_final_test(self, config, episodes_per_test=100, tics_per_action=12, epochs=11):
        self.set_up_game_environment(config, False)

        self.load_model()

        writer = SummaryWriter(comment=('_' + self.model_name + '_final_test'))
        writer.filename_suffix = self.model_name

        for epoch in range(epochs):
            avg_score = 0.0

            for e in trange(episodes_per_test):
                avg_score += self.test_run_fast(tics_per_action)

            avg_score /= episodes_per_test
            writer.add_scalar('Score_epoch_size_' + str(episodes_per_test), avg_score, epoch)
    def start_training(self, config, epoch_count=10, episodes_per_epoch=100, episodes_per_test=100, tics_per_action=12, hardcoded_path=False,
                       fast_train=True, tune_config=None, use_ppo=False):
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
                    actor_loss, critic_loss = self.train_run_fast(tics_per_action, first_run)
                else:
                    actor_loss, critic_loss = self.train_run(tics_per_action, first_run)

                writer.add_scalar('Actor_Loss_epoch_size_' + str(episodes_per_epoch), actor_loss, e + epoch * episodes_per_epoch)
                writer.add_scalar('Critic_Loss_epoch_size_' + str(episodes_per_epoch), critic_loss, e + epoch * episodes_per_epoch)
                writer.add_scalar('Loss_epoch_size_' + str(episodes_per_epoch), actor_loss+critic_loss, e + epoch * episodes_per_epoch)
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


class A2CPPO(A2C):
    def __init__(self, memory_size=10000, model_name='default_A2C_model', learning_rate=1e-4, batch_size=64):
        super().__init__(learning_rate=learning_rate, model_name=model_name)
        self.old_pd = None
        self.max_al = 0.0
        self.max_cl = 0.0
        self.frame_stack_size = 1
        self.min_reward = 10000000.0
        self.max_reward = -10000000.0

    def remember(self, state, action, reward, next_state, done):
        action = self.actions.index(action)
        if reward < self.min_reward:
            self.min_reward = reward
        if reward > self.max_reward:
            self.max_reward = reward

        with torch.no_grad():
            s = copy.copy(state)
            s = np.expand_dims(s, axis=0)
            s = torch.from_numpy(s).float().cpu()
            old_pd = self.actor_model.forward(s)
            old_pd = np.squeeze(old_pd)
            old_pd = old_pd[action]
            old_pd = old_pd.cpu().numpy()

            #advantage =
        # state, action, reward, next state, done, old_policy
        self.memory.append([state, action, reward, next_state, done, old_pd])

    def replay(self, batch=None):
        minibatch = random.sample(self.memory, self.batch_size)

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

        #rewards = (rewards-self.min_reward)/(self.max_reward-self.min_reward)

        row = np.arange(self.batch_size)

        a = row, actions
        v = self.critic_model.forward(states)
        pd = self.actor_model(states)
        with torch.no_grad():
            ns_v = self.critic_model.forward(next_states)

        ns_v = np.squeeze(ns_v)
        v = np.squeeze(v)

        q = rewards + 0.99 * ns_v * not_dones

        advantage = q - v

        pd = pd.squeeze(0)

        eps = 1e-2

        pd_logs = torch.log(pd+eps)
        old_pd_logs = torch.log(old_pds+eps)

        #pd_logs = torch.log(pd)
        with torch.no_grad():
            entropy = pd * pd_logs
            entropy = -torch.sum(entropy, dim=1)
            #pd = pd.squeeze(0)


        entropy_constant = 0.001
        #loss = torch.abs((actor_loss + critic_loss).mean())

        #c1 = 1.0
        #c2 = 0.01


        ratios = torch.exp((pd_logs[a]+1e-10)-(old_pd_logs+1e-10))
        #ratios = torch.exp(pd_logs[a] - old_pd_logs)
        #ratios = pd[a]/old_pds

        eps = 0.2

        s1 = ratios * advantage

        # https://spinningup.openai.com/en/latest/algorithms/ppo.html

        b = torch.clamp(ratios, 1-eps, 1+eps) * advantage
        s2 = b
        surr = -torch.min(s1, s2)


        #b = torch.clamp(advantage, 0, 1)
        #torch.ceil(b)
        #b = b - (1-b)
        #s2 = (1+(eps*b))*advantage

        #actor_loss = -torch.mean(torch.min(s1, s2))

        #critic_loss = torch.mean((advantage)**2)

        #actor_loss = (-pd_logs[a]*advantage)
        actor_loss = -torch.min(s1, s2)
        critic_loss = 0.5 * (advantage).pow(2)

        #loss = torch.abs(actor_loss + critic_loss).mean()
        """
        if self.max_cl < float(critic_loss):
            self.max_cl = float(critic_loss)
            print("max critic loss: ", float(critic_loss))

        if self.max_al < float(actor_loss):
            self.max_al = float(actor_loss)
            print("s1: ", s1)
            print("s2: ", s2)
            print("max actor  loss: ", float(actor_loss))
        """

        loss = torch.abs(actor_loss + critic_loss).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Temp solution, should be calculated using vectors instead
        pd = pd[a]

        for i in range(self.batch_size):
            minibatch_og[i][5] = float(pd[i])

        return actor_loss.mean().item(), critic_loss.mean().item()