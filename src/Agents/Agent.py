import itertools as it
import random
from collections import deque  # for memory

import cv2 as cv2
import numpy as np
import torch
from vizdoom import *


class AgentBase:
    def __init__(self, learning_rate=1e-4, model_name='default_model'):
        self.actions = []
        self.model_name = model_name
        self.model_path = 'models/'+model_name+'.pth'
        self.lr = learning_rate
        self.criterion = None
        self.model = None
        self.optimizer = None
        self.exploration = 1.0
        self.exploration_decay = 0.9995
        self.dr = 0.9  # discount rate
        self.min_exploration = 0.1
        self.downscale = (30, 45)
        self.frame_stack_size = 1
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
            done = game.is_episode_finished()

        return game.get_total_reward()

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
                steps += 1
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
            state = np.array([frame]).astype(np.float32)

            action = self.get_action(state)
            reward = game.make_action(action, tics_per_action)

            done = game.is_episode_finished()

            if not done:
                frame = self.preprocess(game.get_state().screen_buffer)
            else:
                frame = np.zeros(self.downscale).astype(np.float32)

            next_state = np.array([frame]).astype(np.float32)

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
    def get_action(self, state, explore=True):
        return random.choice(self.actions)



