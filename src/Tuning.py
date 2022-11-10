import copy
import numpy as np
import vizdoom
from collections import deque
from functools import partial
from ray import tune


from torch.utils.tensorboard import SummaryWriter

from src import Agent, DoomEnvironment


#def tune_train(game, agent, episodes_per_epoch, config):
def tune_train(config, episodes_per_epoch=1000):
    #writer = SummaryWriter()
    #agent.load_model()
    agent = Agent.AgentDuelDQN(model_name='DDQN')
    doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/basic.cfg", agent, hardcoded_path=True)
    game = doomEnv.game
    agent.set_model_configuration(config)
    print("Detect change")

    total_loss = 0
    tics_per_action = 6
    total_reward = 0
    scores = deque([], maxlen=100)
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

            loss += agent.train(frames, action, next_state, reward, done)

        scores.append(game.get_total_reward())
        total_reward += game.get_total_reward()
        total_loss += loss

        mean_score = sum(scores)/len(scores)
        mean_loss = total_loss/(e+1)
        tune.report(mean_score=mean_score, mean_loss=mean_loss, exploration=agent.exploration)

        #writer.add_scalar('Score', game.get_total_reward(), e)
        #writer.add_scalar('Exploration', agent.exploration, e)
        #writer.add_scalar('Loss', loss, e)

        agent.decay_exploration()

    #mean_score = total_reward/episodes_per_epoch
    #mean_loss = total_loss/episodes_per_epoch

    #tune.report(mean_score, mean_loss)
    #agent.save_model()
    #return {"mean_score": mean_score}


def run_tuning(episodes_per_epoch, num_samples=10, max_num_epochs=10):
    config = {
        "c1": tune.sample_from(lambda _: 2 ** np.random.randint(3, 6)),
        "c2": tune.sample_from(lambda _: 2 ** np.random.randint(3, 6)),
        "c3": tune.sample_from(lambda _: 2 ** np.random.randint(3, 6)),
        "c4": tune.sample_from(lambda _: 2 ** np.random.randint(3, 6))
    }

    scheduler = tune.schedulers.ASHAScheduler(
        metric="mean_score",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = tune.CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["mean_score", "mean_loss", "exploration", "training_iteration"])

    #partial(tune_train, game, agent, episodes_per_epoch),
    result = tune.run(
        partial(tune_train),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_result = result.get_best_trial("mean_score", "max", "last")
    print("Best trial config: {}".format(best_result.config))
    print("Best final mean loss: {}".format(best_result.last_result["mean_loss"]))
    print("Best result final mean score: {}".format(best_result.last_result["mean_score"]))

    #print(result.best_config)