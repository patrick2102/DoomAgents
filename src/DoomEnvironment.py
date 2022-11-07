from collections import deque

from vizdoom import *
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import torch
import copy


class DoomEnvironmentInstance:
    def __init__(self, config, agent):
        self.agent = agent
        self.game = DoomGame()
        self.game.load_config(config)
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

    def run(self, episode_count):
        game = self.game
        agent = self.agent

        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        actions = [shoot, left, right]
        agent.set_available_actions(actions)

        total_time = 0
        tics_per_action = 12

        for e in range(episode_count):
            game.new_episode()
            start = time.time()
            done = False
            step_count = 0

            state = agent.get_image(game.get_state())
            action = agent.get_action(state)
            reward = game.make_action(action)

            while not done:
                #print(reward)
                next_state = agent.get_image(game.get_state())

                if step_count % tics_per_action == 0:
                    next_action = agent.get_action(state)
                    next_reward = game.make_action(next_action)
                else:
                    next_action = action
                    game.advance_action()
                    next_reward = game.get_last_reward()

                done = game.is_episode_finished()

                agent.train(state, action, next_state, reward, done)

                state = next_state
                reward = next_reward
                action = next_action

                step_count += 1

            agent.decay_exploration()

            end = time.time()
            total_time += end-start
            print("average episode time: ", total_time/(e+1))
            print("Result:", game.get_total_reward())
            time.sleep(0.1)

    def run_statistics(self, episodes_per_epoch, epoch_count):
        game = self.game
        agent = self.agent

        agent.load_model()

        total_time = 0

        total_reward = 0

        tics_per_action = 12

        scores = deque([], maxlen=100)

        for epoch in range(epoch_count):
            plt.ion()
            plt.show()
            plt.xlabel('Episode')
            plt.ylabel('Average reward')
            plt.title('Doom')
            x = []
            rewards = []
            losses = []
            times = []

            print("epoch: ", epoch)
            for e in range(episodes_per_epoch):
                game.new_episode()
                avg_loss = 0
                step_count = 0
                start = time.time()
                done = False

                while not done:
                    state = game.get_state()
                    action = agent.get_action(state)
                    reward = game.make_action(action)

                    done = game.is_episode_finished()

                    for i in range(tics_per_action):
                        if done:
                            break
                        game.advance_action()
                        done = game.is_episode_finished()

                    if not done:
                        next_state = game.get_state()
                    else:
                        next_state = None

                    avg_loss += agent.train(state, action, next_state, reward, done)

                    step_count += 1

                end = time.time()

                agent.decay_exploration()

                #total_time += (end-start)/step_count
                avg_time = (end-start)/step_count
                total_reward += game.get_total_reward()
                scores.append(game.get_total_reward())
                avg_reward = sum(scores)/len(scores)
                #print("average episode time: ", avg_time)
                #print("Result:", game.get_total_reward())
                #print("exploration rate: ", agent.exploration)
                #avg_loss /= step_count
                x.append(e)
                rewards.append(avg_reward)
                losses.append(int(avg_loss))
                times.append(avg_time)
                plt.plot(x, rewards)
                #plt.plot(x, times)
                plt.draw()
                plt.pause(0.1)

                time.sleep(0.1)
            plt.clf()
            rewards.clear()
            x.clear()
            agent.save_model()
