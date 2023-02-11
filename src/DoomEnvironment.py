from vizdoom import DoomGame
import itertools as it
from src.Agents import Agent
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


def SetUpGameEnvironment(config):
        # Set up game environment
        game = DoomGame()

        game.load_config(config)

        game.add_game_args("-host 2 "
                           # This machine will function as a host for a multiplayer game with this many players (including this machine). 
                           # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                           "-port 5029 "  # Specifies the port (default is 5029).
                           "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
                           "-deathmatch "  # Deathmatch rules are used for the game.
                           "+timelimit 10.0 "  # The game (episode) will end after this many minutes have elapsed.
                           "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
                           "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
                           "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
                           "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
                           "+sv_nocrouch 1 "  # Disables crouching.
                           "+viz_respawn_delay 10 "  # Sets delay between respawns (in seconds, default is 0).
                           "+viz_nocheat 1")  # Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

        game.init()

        # Set up model and possible actions
        n = game.get_available_buttons_size()

        actions = [list(a) for a in it.product([0, 1], repeat=n)]
        return game, actions

def player1(agent: Agent.AgentBase, config, episodes):
    game = DoomGame()

    game.load_config(config)
    game.add_game_args("-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1")
    game.add_game_args("+name Player1 +colorset 0")

    game.init()

    n = game.get_available_buttons_size()

    agent.actions = [list(a) for a in it.product([0, 1], repeat=n)]

    for i in range(episodes):

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            game.make_action(agent.get_action())

        print("Episode finished!")
        print("Player1 frags:", game.get_game_variable(vzd.GameVariable.FRAGCOUNT))

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.
        game.new_episode()

    game.close()

def StartMultiplayerMatchTrain(agent1: Agent.AgentBase, agent2: Agent.AgentBase, config, epoch_count=10,
                               episodes_per_epoch=1000, episodes_per_test=10):

    # Set up game environment and actions
    game, actions = SetUpGameEnvironment(config)

    #Set up agents
            #self.load_model_config(tune_config)
    agent1.load_model()
    agent2.load_model()

    # Set up ray and training details
    writer1 = SummaryWriter(comment=('_multi_' + agent1.model_name))
    writer2 = SummaryWriter(comment=('_multi_' + agent2.model_name))
    writer1.filename_suffix = agent1.model_name + '_vs_' + agent2.model_name
    writer2.filename_suffix = agent1.model_name + '_vs_' + agent2.model_name
    first_run = False

    # Epoch runs a certain amount of episodes, followed a test run to show performance.
    # At the end the model is saved on disk
    for epoch in range(epoch_count):
        print("epoch: ", epoch + 1)

        for e in trange(episodes_per_epoch):
            actor_loss, critic_loss = self.train_run_fast(tics_per_action, first_run)

            writer1.add_scalar('Actor_Loss_epoch_size_' + str(episodes_per_epoch), actor_loss,
                              e + epoch * episodes_per_epoch)
            writer1.add_scalar('Critic_Loss_epoch_size_' + str(episodes_per_epoch), critic_loss,
                              e + epoch * episodes_per_epoch)
            writer1.add_scalar('Loss_epoch_size_' + str(episodes_per_epoch), actor_loss + critic_loss,
                              e + epoch * episodes_per_epoch)
            writer1.add_scalar('Reward_epoch_size_' + str(episodes_per_epoch), game.get_total_reward(),
                              e + epoch * episodes_per_epoch)
            writer1.add_scalar('Exploration_epoch_size_' + str(episodes_per_epoch), agent1.exploration,
                              e + epoch * episodes_per_epoch)

            writer2.add_scalar('Actor_Loss_epoch_size_' + str(episodes_per_epoch), actor_loss,
                              e + epoch * episodes_per_epoch)
            writer2.add_scalar('Critic_Loss_epoch_size_' + str(episodes_per_epoch), critic_loss,
                              e + epoch * episodes_per_epoch)
            writer2.add_scalar('Loss_epoch_size_' + str(episodes_per_epoch), actor_loss + critic_loss,
                              e + epoch * episodes_per_epoch)
            writer2.add_scalar('Reward_epoch_size_' + str(episodes_per_epoch), game.get_total_reward(),
                              e + epoch * episodes_per_epoch)
            writer2.add_scalar('Exploration_epoch_size_' + str(episodes_per_epoch), agent2.exploration,
                              e + epoch * episodes_per_epoch)

        agent1.save_model()
        agent2.save_model()
        avg_score = 0.0
        for e in trange(episodes_per_test):
            avg_score += self.test_run_fast(tics_per_action)

        avg_score /= episodes_per_test
        writer1.add_scalar('Score_epoch_size_' + str(episodes_per_test), avg_score, epoch)
        writer2.add_scalar('Score_epoch_size_' + str(episodes_per_test), avg_score, epoch)