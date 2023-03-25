import itertools as it
import random
from collections import deque  # for memory
from time import sleep, time

import cv2 as cv2
import numpy as np
import torch
from vizdoom import *
from vizdoom.vizdoom import DoomGame
import vizdoom as viz


class AgentBase:
    def __init__(self, learning_rate=0.0001, model_name='default_model'):
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
        self.downscale_1 = (1, 30, 45)
        self.frame_stack_size = 1
        self.game = None

    def set_model_path(self, model_name):
        self.model_path = 'models/'+model_name+'.pth'

    def preprocess(self, state):
        s = state

        s = cv2.resize(s, self.downscale, interpolation=cv2.INTER_AREA)

        s = np.moveaxis(s, 1, 0)
        #s = np.expand_dims(s, axis=0)

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

    def set_up_game_async(self, config):
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(True)
        self.game.set_mode(viz.Mode.ASYNC_PLAYER)
        self.game.init()

        # Set up model and possible actions
        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]

    def set_available_actions(self, avail_actions):
        self.actions = avail_actions

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

    def train(self, state, last_action, next_state, reward,  done=False):
        raise NotImplementedError


    def run_async_test(self, config):

        self.set_up_game_environment(config, False)
        self.load_model()

        episodes_to_watch = 10
        tics_per_action = 12

        for i in range(episodes_to_watch):
            self.game.new_episode("replays/episode" + str(i) + "_rec.lmp")
            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                best_action = self.get_action(state, explore=False)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(best_action)
                for _ in range(tics_per_action):
                    self.game.advance_action()
                # reward = self.game.make_action(best_action)


            # Sleep between episodes
            sleep(1.0)
            score = self.game.get_total_reward()
            print("Total score: ", score)
        self.game.close()

    def replay_show(self, config):

        episodes = 10
        game = DoomGame()
        game.load_config(config)
        game.set_screen_resolution(ScreenResolution.RES_800X600)
        game.set_render_hud(True)

        # Replay can be played in any mode.
        game.set_mode(Mode.SPECTATOR)

        game.init()

        for i in range(episodes):

            # Replays episodes stored in given file. Sending game command will interrupt playback.
            game.replay_episode("replays/episode" + str(i) + "_rec.lmp")

            while not game.is_episode_finished():
                s = game.get_state()

                # Use advance_action instead of make_action.
                game.advance_action()

                r = game.get_last_reward()
                # game.get_last_action is not supported and don't work for replay at the moment.

                # print("State #" + str(s.number))
                # print("Game variables:", s.game_variables[0])
                # print("Reward:", r)
                # print("=====================")

            # print("Episode", i, "finished.")
            # print("total reward:", game.get_total_reward())
            # print("************************")

        game.close()

    """Run without multiple images and exploration"""
    def test_run_fast(self, tics_per_action):
        game = self.game
        game.new_episode()
        done = False
        prev_frames = deque([np.zeros(self.downscale).astype(np.float32)] * self.frame_stack_size,
                            maxlen=self.frame_stack_size)

        while not done:
            frame = self.preprocess(game.get_state().screen_buffer)
            #prev_frames.append(frame)
            #state = np.array(prev_frames)
            state = frame
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

            game.make_action(action, tics_per_action)

            done = game.is_episode_finished()

        return game.get_total_reward()

    """
        Train model
    """
    def train_run(self, tics_per_action, first_run):
        game = self.game
        game.new_episode()
        prev_frames = deque([np.zeros(self.downscale_1).astype(np.float32)] * self.frame_stack_size,
                            maxlen=self.frame_stack_size)
        done = False
        loss = 0
        steps = 0

        while not done:
            steps += 1
            frame = self.preprocess(game.get_state().screen_buffer)
            prev_frames.append(frame)

            state = np.array(prev_frames)
            action = self.get_action(frame)
            game.set_action(action)
            reward = 0

            for i in range(tics_per_action):
                game.advance_action()
                steps += 1
                reward += game.get_last_reward()
                done = game.is_episode_finished()
                if done:
                    frame = np.zeros(self.downscale_1).astype(np.float32)
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

            loss += self.train(state, action, next_state, reward, done)

            if not first_run:
                self.decay_exploration()

        loss /= steps

        return loss

    def load_model_config(self, tune_config):
        raise NotImplementedError

    def load_model(self):
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



