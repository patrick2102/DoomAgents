from src import DoomEnvironment, Agent, Models, EvolutionaryAgents, Tuning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#agent = Agent.AgentRandom()
#doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/basic.cfg", agent)

#doomEnv.run(episode_count=10)
"""

# DQN agent:
agentDQN = Agent.AgentDQN()
criterion = nn.MSELoss()
x_size = 320
y_size = 240
img_size = int(x_size*y_size)*3
action_space = 3
N = 2000
model = Models.SimpleLinearNN(x_size, y_size, action_space)
agentDQN.set_model(criterion, model, N)
#model_conv = Models.SimpleConvNN(32, 24, action_size)
#agentDQN.set_model(criterion, model_conv, N)

#DDQN agent:
agentDDQN = Agent.AgentDoubleDQN()


#model_dqn = Models.SimpleConvNN(x_size, y_size, action_space)

model3 = Models.ConvLinearNN(x_size, y_size, action_space)

#agentDDQN.init_model(criterion, model_dqn)
#agentDDQN.init_model(criterion, model)
agentDDQN.init_model(criterion, model3)

#print(torch.cuda.is_available())
doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/basic.cfg", agentDDQN)
#doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/basic.cfg", agentDQN)
#doomEnv.run(episode_count=1000)
doomEnv.run_statistics(episode_count=10)

agentDDQN.save_model()

doomGame = doomEnv.get_game()

left = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]
actions = [shoot, left, right]

#evolutionaryAgent = EvolutionaryAgents.EvolutionaryAgentSimple(x_size, y_size, actions)

#evolutionaryAgent.train(10, 10, doomGame)

"""
"""
def start_training_ddqn(n=100):
    criterion = nn.MSELoss()
    x_size = 320
    y_size = 240
    img_size = int(x_size * y_size) * 3
    action_space = 3

    agentDDQN = Agent.AgentDoubleDQN()
    #model = Models.ConvLinearNN(x_size, y_size, action_space)
    #model = Models.SimpleLinearNN(x_size, y_size, action_space)
    model = Models.ConvLinearNN2(x_size, y_size, action_space)
    agentDDQN.init_model(criterion, model)

    doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/basic.cfg", agentDDQN)
    doomEnv.run_statistics(episode_count=n)

    agentDDQN.save_model()

def resume_training_ddqn(n=100):
    criterion = nn.MSELoss()
    x_size = 320
    y_size = 240
    img_size = int(x_size * y_size) * 3
    action_space = 3

    agentDDQN = Agent.AgentDoubleDQN()
    #model = Models.ConvLinearNN(x_size, y_size, action_space)
    #model = Models.SimpleLinearNN(x_size, y_size, action_space)
    model = Models.ConvLinearNN2(x_size, y_size, action_space)
    agentDDQN.init_model(criterion, model)

    agentDDQN.load_model()

    doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/basic.cfg", agentDDQN)
    doomEnv.run_statistics(episode_count=n)

    agentDDQN.save_model()
    print("model saved")

start_training_ddqn(1)
resume_training_ddqn(1000)
"""

def train_dqn():
    episodes_per_epoch = 1000
    epochs = 10
    #agentDQN = Agent.AgentDQN(model_name='DQN_framestack')
    agentDQN = Agent.AgentDuelDQN(model_name='DDQN_Opt5')
    doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/simpler_basic.cfg", agentDQN)
    doomEnv.run_statistics(episodes_per_epoch=episodes_per_epoch, epoch_count=epochs)

def tune_dueldqn():
    episodes_per_epoch = 100
    samples = 10
    epochs = 10
    #agentDQN = Agent.AgentDuelDQN(model_name='DDQN')
    #doomEnv = DoomEnvironment.DoomEnvironmentInstance("scenarios/basic.cfg", agentDQN)
    Tuning.run_tuning(episodes_per_epoch, samples, epochs)
    #Tuning.tune_learning_rate(episodes_per_epoch, samples, epochs)

#tune_dueldqn()

#train_dqn()

#newAgent = Agent.AgentNew(model_name='DoubleDuelDQN')
#newAgent.start_training("scenarios/basic.cfg")


#agentDQN = Agent.AgentDQN(model_name='DDQN')
#agentDQN.start_training("scenarios/simpler_basic.cfg")
#agentDQN.start_training("scenarios/simpler_basic.cfg")

#agentDuelDQN = Agent.AgentDuelDQN()
#agentDuelDQN.start_training("scenarios/simpler_basic.cfg")

agentDoubleDuelDQN = Agent.AgentDoubleDuelDQN()
agentDoubleDuelDQN.start_training("scenarios/basic.cfg", epoch_count=100)


#agentDoubleDuelDQN = Agent.AgentDoubleDuelDQN(model_name='DoubleDuelDQN2')
#agentDoubleDuelDQN.start_training("scenarios/basic.cfg", epoch_count=100)

#while True:
#    continue

