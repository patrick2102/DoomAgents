#!/usr/bin/python3

#####################################################################
# This script presents how to use Doom's native demo mechanism to
# record multiplayer game and replay it with perfect accuracy.
#####################################################################

# WARNING:
# Due to the bug in build-in bots recording game with bots will result in the desynchronization of the recording.

from multiprocessing import Process
import os
from random import choice
from vizdoom import *
import Agent
import itertools as it
import numpy as np
from collections import deque

total_episodes = 100


def player1():
    agentDQN = Agent.AgentDQN(model_name='DoubleDuelDQN_simple_dm01')

    agentDQN.load_model()

    game = DoomGame()

    game.load_config("C:/Users/Lukas/Documents/GitHub/DoomAgents/scenarios/multi_duel.cfg")
    game.add_game_args("-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1 ")
    game.add_game_args("+name Player1 +colorset 0")

    # Unfortunately multiplayer game cannot be recorded using new_episode() method, use this command instead.
    game.add_game_args("-record multi_rec.lmp")

    game.init()

    n = game.get_available_buttons_size()
    agentDQN.actions = [list(a) for a in it.product([0, 1], repeat=n)]

    for i in range(total_episodes):
        
        loss = 0

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            frame = agentDQN.preprocess(game.get_state().screen_buffer)
            state = np.array([frame] * agentDQN.frame_stack_size).astype(np.float32)
            action = agentDQN.get_action(state)

            game.make_action(action, 8)

            done = game.is_episode_finished()
            reward = game.get_game_variable(
                GameVariable.FRAGCOUNT)  # Manually calculate reward based on frags (and maybe ammo? health?)

            if not done:
                frame = agentDQN.preprocess(game.get_state().screen_buffer)
            else:
                frame = np.zeros(agentDQN.downscale).astype(np.float32)

            next_state = np.array([frame] * agentDQN.frame_stack_size).astype(np.float32)

            loss += agentDQN.train(state, action, next_state, reward, done)

        print("Game finished!")
        print("Player1 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
        game.new_episode()

    game.close()


def player2():
    game = DoomGame()

    game.load_config("C:/Users/Lukas/Documents/GitHub/DoomAgents/scenarios/multi_duel.cfg")
    game.set_window_visible(False)
    game.add_game_args("-join 127.0.0.1")
    game.add_game_args("+name Player2 +colorset 3")

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    for i in range(total_episodes):
        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            game.make_action(choice(actions))

        # print("Game finished!")
        # print("Player2 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
        game.new_episode()
    game.close()


if __name__ == '__main__':
    print("\nRECORDING")
    print("************************\n")

    p1 = Process(target=player1)
    p1.start()

    player2()
