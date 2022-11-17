import copy
import numpy as np
import vizdoom
from collections import deque
from functools import partial
from ray import tune


from torch.utils.tensorboard import SummaryWriter

from src import Agent, DoomEnvironment


#def tune_train(game, agent, episodes_per_epoch, config):
def tune_train(config, episodes_per_epoch=1000, epoch_count=10):
    #writer = SummaryWriter()
    #agent.load_model()
    agent = Agent.AgentDuelDQN(model_name='DDQN')
    doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/basic.cfg", agent, hardcoded_path=True)
    game = doomEnv.game
    agent.set_model_configuration(config)
    #print("Detect change")

    tics_per_action = 6
    #scores = deque([], maxlen=100)
    #losses = deque([], maxlen=100)
    for epoch in range(epoch_count):
        total_reward = 0
        total_loss = 0
        for e in range(episodes_per_epoch):
            game.new_episode()
            done = False
            loss = 0
            prev_frames = deque([None, None, None, None], maxlen=4)

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

                total_loss += agent.train(frames, action, next_state, reward, done)

            total_reward += game.get_total_reward()
            agent.decay_exploration()

        if epoch != 0:
            mean_score = total_reward/episodes_per_epoch
            mean_loss = total_loss/episodes_per_epoch
            tune.report(mean_score=mean_score, mean_loss=mean_loss, exploration=agent.exploration)
            agent.update_target_model()

def tune_learning_rate(episodes_per_epoch, num_samples=10, max_num_epochs=10):
    config = {
        "c1": 8,
        "c2": 8,
        "c3": 8,
        "c4": 8,
        "momentum": 0,
        "lr": tune.loguniform(1e-6, 1e-1)
    }

    scheduler = tune.schedulers.ASHAScheduler(
        metric="mean_score",
        mode="max",
        max_t=episodes_per_epoch,
        grace_period=1,
        reduction_factor=2
    )

    reporter = tune.CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["mean_score", "mean_loss", "exploration", "training_iteration"])

    #partial(tune_train, game, agent, episodes_per_epoch),
    result = tune.run(
        partial(tune_train, episodes_per_epoch=episodes_per_epoch),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)



def run_tuning(episodes_per_epoch, num_samples=10, max_num_epochs=10):
    config = {
        "c1": tune.sample_from(lambda _: 2 ** np.random.randint(3, 6)),
        "c2": tune.sample_from(lambda _: 2 ** np.random.randint(3, 6)),
        "c3": tune.sample_from(lambda _: 2 ** np.random.randint(3, 6)),
        "c4": tune.sample_from(lambda _: 2 ** np.random.randint(3, 6)),
        "momentum": 0.9,
        "lr": tune.choice([1e-5])
    }

    scheduler = tune.schedulers.ASHAScheduler(
        metric="mean_score",
        mode="max",
        max_t=episodes_per_epoch,
        grace_period=1,
        reduction_factor=2
    )

    reporter = tune.CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["mean_score", "mean_loss", "exploration", "training_iteration"])

    #partial(tune_train, game, agent, episodes_per_epoch),
    result = tune.run(
        partial(tune_train, episodes_per_epoch=episodes_per_epoch, epoch_count=max_num_epochs),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_result = result.get_best_trial("mean_score", "max", "last")
    print("Best trial config: {}".format(best_result.config))
    print("Best final mean loss: {}".format(best_result.last_result["mean_loss"]))
    print("Best result final mean score: {}".format(best_result.last_result["mean_score"]))

    #print(result.best_config)