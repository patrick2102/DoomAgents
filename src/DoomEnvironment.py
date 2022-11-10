from collections import deque
from torch.utils.tensorboard import SummaryWriter
from vizdoom import *
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import torch
import copy


class DoomEnvironmentInstance:
    def __init__(self, config, agent, hardcoded_path=False):
        self.agent = agent
        self.game = DoomGame()
        if hardcoded_path:
            config_path = "C:/Uni/3rd_Semester/DeepLearning/Project/DoomAgents/" + config
        else:
            config_path = config
        self.game.load_config(config_path)
        self.game.init()
        self.downscale = (30, 45)
        self.ticrate = 35
        self.game.set_ticrate(self.ticrate)
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        actions = [shoot, left, right]
        agent.set_available_actions(actions)

    def get_game(self):
        return self.game

    def get_state_as_image(self, state):
        s = state.screen_buffer

        s = np.moveaxis(s, 0, 2)

        s = cv2.resize(s, self.downscale, interpolation=cv2.INTER_AREA)
        s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)

        s = np.moveaxis(s, 1, 0)

        s = np.array(s, dtype=float)/255

        s = np.expand_dims(s, axis=0)

        return s

    def run_episode(self, game, agent, tics_per_action=12):
        game.new_episode()
        done = False
        prev_frames = deque([None, None, None, None], maxlen=4)
        loss = 0

        while not done:
            game_state = copy.deepcopy(game.get_state().screen_buffer)
            frames = [game_state] + copy.deepcopy(list(prev_frames))
            action = agent.get_action(frames)
            reward = game.make_action(action)
            done = game.is_episode_finished()

            for i in range(tics_per_action):
                if done:
                    break
                game.advance_action()
                done = game.is_episode_finished()

                if done:
                    break

                game_state = copy.deepcopy(game.get_state().screen_buffer)
                prev_frames.append(game_state)
                done = game.is_episode_finished()

            if not done:
                next_state = copy.deepcopy(game.get_state().screen_buffer)
            else:
                next_state = None

            next_state = [next_state] + copy.deepcopy(list(prev_frames))

            loss += agent.train(frames, action, next_state, reward, done)
        return loss, 0

    def run_epoch(self, game, agent, episode_count, tics_per_action=12):
        for e in range(episode_count):
            yield self.run_episode(game, agent, tics_per_action=tics_per_action)

    """
    def run(self, epoch_count, episode_count):
        game = self.game
        agent = self.agent
        agent.load_model()
        tics_per_action = 12

        for epoch in range(epoch_count):
            for e in range(episodes_per_epoch):
                game.new_episode()
                running_loss = 0
                step_count = 0
                start = time.time()
                done = False
                prev_frames = deque([None, None, None, None], maxlen=4)

                while not done:
                    game_state = copy.deepcopy(game.get_state().screen_buffer)
                    frames = [game_state] + copy.deepcopy(
                        list(prev_frames))  # make sure list doesn't get updated in loop below
                    action = agent.get_action(frames)
                    reward = game.make_action(action)

                    done = game.is_episode_finished()

                    for i in range(tics_per_action):
                        if done:
                            break
                        game.advance_action()
                        done = game.is_episode_finished()

                        if done:
                            break

                        game_state = copy.deepcopy(game.get_state().screen_buffer)
                        prev_frames.append(game_state)
                        done = game.is_episode_finished()

                    if not done:
                        next_state = copy.deepcopy(game.get_state().screen_buffer)
                    else:
                        next_state = None

                    next_state = [next_state] + copy.deepcopy(list(prev_frames))

                    loss = agent.train(frames, action, next_state, reward, done)

                    if loss != -1:
                        running_loss += loss

                    step_count += 1

                end = time.time()

                agent.decay_exploration()

                # total_time += (end-start)/step_count
                avg_time = (end - start) / step_count
                total_reward += game.get_total_reward()
                scores.append(game.get_total_reward())
                avg_reward = sum(scores) / len(scores)
                writer.add_scalar('Score', game.get_total_reward(), e)
                writer.add_scalar('Exploration', agent.exploration, e)
                writer.add_scalar('Loss', running_loss, e)

                time.sleep(0.1)

            scores.clear()

            agent.save_model()
    """

    def run_statistics(self, episodes_per_epoch, epoch_count):
        game = self.game
        agent = self.agent
        writer = SummaryWriter()

        agent.load_model()

        total_time = 0

        total_reward = 0

        tics_per_action = 12

        scores = deque([], maxlen=100)

        for epoch in range(epoch_count):

            print("epoch: ", epoch)
            for e in range(episodes_per_epoch):
                game.new_episode()
                running_loss = 0
                step_count = 0
                start = time.time()
                done = False
                prev_frames = deque([None, None, None, None], maxlen=4)

                while not done:
                    game_state = copy.deepcopy(game.get_state().screen_buffer)
                    frames = [game_state] + copy.deepcopy(list(prev_frames))  # make sure list doesn't get updated in loop below
                    action = agent.get_action(frames)
                    reward = game.make_action(action)

                    done = game.is_episode_finished()

                    for i in range(tics_per_action):
                        if done:
                            break
                        game.advance_action()
                        done = game.is_episode_finished()

                        if done:
                            break

                        game_state = copy.deepcopy(game.get_state().screen_buffer)
                        prev_frames.append(game_state)
                        done = game.is_episode_finished()

                    if not done:
                        next_state = copy.deepcopy(game.get_state().screen_buffer)
                    else:
                        next_state = None

                    next_state = [next_state] + copy.deepcopy(list(prev_frames))

                    loss = agent.train(frames, action, next_state, reward, done)

                    if loss != -1:
                        running_loss += loss

                    step_count += 1

                end = time.time()

                agent.decay_exploration()

                #total_time += (end-start)/step_count
                avg_time = (end-start)/step_count
                total_reward += game.get_total_reward()
                scores.append(game.get_total_reward())
                avg_reward = sum(scores)/len(scores)
                writer.add_scalar('Score', game.get_total_reward(), e)
                writer.add_scalar('Exploration', agent.exploration, e)
                writer.add_scalar('Loss', running_loss, e)

                time.sleep(0.1)

            scores.clear()

            agent.save_model()