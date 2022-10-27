from vizdoom import *
import random
import time
import matplotlib.pyplot as plt
import torch


class DoomEnvironmentInstance:
    def __init__(self, config, agent):
        self.agent = agent
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.init()

    def get_game(self):
        return self.game

    def run(self, episode_count):
        game = self.game
        agent = self.agent

        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        actions = [shoot, left, right]
        agent.set_available_actions(actions)

        total_time = 0

        for e in range(episode_count):
            game.new_episode()
            start = time.time()
            while not game.is_episode_finished():
                state = game.get_state()
                action = agent.get_action(state)
                #print(action)
                reward = game.make_action(action)
                agent.train(state, reward, game.is_episode_finished())

                #print("\treward:", reward)
                #time.sleep(0.02)

            agent.decay_exploration()

            end = time.time()
            total_time += end-start
            print("average episode time: ", total_time/(e+1))
            print("Result:", game.get_total_reward())
            time.sleep(0.1)

    def run_statistics(self, episode_count):
        plt.ion()
        plt.show()
        plt.xlabel('Episode')
        plt.ylabel('Average reward')
        plt.title('Doom')
        x = []
        rewards = []
        losses = []
        times = []

        game = self.game
        agent = self.agent

        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        actions = [shoot, left, right]
        agent.set_available_actions(actions)

        total_time = 0

        total_reward = 0

        for e in range(episode_count):
            game.new_episode()
            avg_loss = 0
            step_count = 0
            start = time.time()
            while not game.is_episode_finished():
                step_count += 1
                state = game.get_state()
                action = agent.get_action(state)
                reward = game.make_action(action)
                next_state = game.get_state()
                avg_loss += agent.train(next_state, reward, game.is_episode_finished())

            agent.decay_exploration()

            end = time.time()
            total_time += end-start
            avg_time = total_time/(e+1)
            total_reward += game.get_total_reward()
            avg_reward = total_reward/(e+1)
            print("average episode time: ", avg_time)
            print("Result:", game.get_total_reward())
            print("exploration rate: ", agent.exploration)
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
