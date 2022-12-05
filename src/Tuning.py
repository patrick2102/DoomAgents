from functools import partial
from ray import tune
from tqdm import trange

from src.Agents import Agent


#def tune_train(game, agent, episodes_per_epoch, config):

def start_tuning(tune_config, agent: Agent.AgentBase, episodes_per_epoch, episodes_per_test, epoch_count, doom_config, tics_per_action=12):

    agent.set_up_game_environment(doom_config, hardcoded_path=True)
    agent.load_model_config(tune_config=tune_config)

    first_run = False

    # Epoch runs a certain amount of episodes, followed a test run to show performance.
    # At the end the model is saved on disk
    for epoch in range(epoch_count):
        print("epoch: ", epoch + 1)
        mean_score = 0.0
        mean_reward = 0.0
        mean_loss = 0.0
        mean_exploration = 0.0

        for e in trange(episodes_per_epoch):
            loss = agent.train_run(tics_per_action, first_run)
            mean_loss += loss

        #agent.save_model()

        for e in trange(episodes_per_test):
            score = agent.test_run(tics_per_action)
            mean_score += score

        mean_score /= episodes_per_test
        mean_loss /= episodes_per_epoch
        first_run = False
        if epoch != 0:
            tune.report(mean_score=mean_score, mean_loss=mean_loss, exploration=agent.exploration)

"""
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
"""

def run_tuning(agent, episodes_per_epoch, doom_config, tune_config, num_samples=10, max_num_epochs=10, episodes_per_test=10):

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
        partial(start_tuning, agent=agent, episodes_per_epoch=episodes_per_epoch, episodes_per_test=episodes_per_test,
                epoch_count=max_num_epochs, doom_config=doom_config),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_result = result.get_best_trial("mean_score", "max", "last")
    print("Best trial config: {}".format(best_result.config))
    print("Best final mean loss: {}".format(best_result.last_result["mean_loss"]))
    print("Best result final mean score: {}".format(best_result.last_result["mean_score"]))

    #print(result.best_config)

