from vizdoom import *
import random
import time
import matplotlib.pyplot as plt
import torch
import copy


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
        tics_per_action = 6

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

        tics_per_action = 6

        for e in range(episode_count):
            game.new_episode()
            avg_loss = 0
            step_count = 0
            start = time.time()
            done = False

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

                avg_loss += agent.train(state, action, next_state, reward, done)

                state = next_state
                reward = next_reward
                action = next_action

                step_count += 1

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
