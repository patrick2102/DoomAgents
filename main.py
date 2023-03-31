import sys

from src.Agents import AgentsA2C
import cProfile

commands = {
    "model": "",
    "scenario": ""
}

def interpretCommands():
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-m':
            commands['model'] = sys.argv[i+1]
            i += 1
        elif sys.argv[i] == '-s':
            commands['scenario'] = sys.argv[i+1]
            i += 1



interpretCommands()

#agent = AgentsA2C.A2CPPO(model_name=commands['model'], batch_size=256)
#agent.start_training(commands['scenario'], episodes_per_test=100, episodes_per_epoch=100, fast_train=False)

agent = AgentsA2C.A2CPPOLSTM(model_name=commands['model'], batch_size=256)
agent.start_training(commands['scenario'], episodes_per_test=100, episodes_per_epoch=100, epoch_count=10, fast_train=False, render=False)

#cProfile.run('agent.start_training(commands[\'scenario\'], episodes_per_test=100, episodes_per_epoch=100, epoch_count=1, fast_train=False)')