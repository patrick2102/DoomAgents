from src.Agents import AgentsDQN, AgentsA2C
from src import DoomEnvironment
import sys

"""running an agent"""

#agentDuelDQN_Basic = AgentsDQN.AgentDuelDQN(model_name='EXAMPLE_MODEL_NAME')
#agentDuelDQN_Basic.start_training("scenarios/basic.cfg")

#agentDuelDQN_HealthGather = AgentsDQN.AgentDuelDQN(model_name='')
#agentDuelDQN_HealthGather.start_training("scenarios/health_gathering.cfg")

# agentDoubleDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDoubleDuelDQN(model_name='')
# agentDoubleDuelDQN_HealthGatherSupreme.run_final_test("scenarios/health_gathering_supreme.cfg")

# agentA2C_DeadlyCorridor = AgentsA2C.A2C(model_name='')
# agentA2C_DeadlyCorridor.run_final_test("scenarios/deadly_corridor.cfg")


"""Basic experiments"""
# agentDQN_Basic = AgentsDQN.AgentDQN(model_name='DQN_Basic')
# agentDQN_Basic.start_training("scenarios/basic.cfg")
# agentDQN_Basic.run_async_test("scenarios/basic.cfg")
# agentDQN_Basic.replay_show("scenarios/basic.cfg")

#agentDuelDQN_Basic = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_Basic')
#agentDuelDQN_Basic.start_training("scenarios/basic.cfg")

#agentDoubleDuelDQN_Basic = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_Basic')
#agentDoubleDuelDQN_Basic.start_training("scenarios/basic.cfg")


"""Health Gather experiments"""

# agentDQN_HealthGather = AgentsDQN.AgentDQN(model_name='DQN_HealthGather')
# agentDQN_HealthGather.run_final_test("scenarios/health_gathering.cfg")
# agentDQN_HealthGather.start_training("scenarios/health_gathering.cfg")

#agentDuelDQN_HealthGather = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_HealthGather')
#agentDuelDQN_HealthGather.start_training("scenarios/health_gathering.cfg")

#agentDoubleDuelDQN_HealthGather = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_HealthGather')
#agentDoubleDuelDQN_HealthGather.start_training("scenarios/health_gathering.cfg")

# agentDQN_HealthGather = AgentsA2C.A2C(model_name='A2C_HealthGather_3k_1_Nodrop')
# agentDQN_HealthGather.start_training("scenarios/health_gathering.cfg")
# agentDQN_HealthGather.run_final_test("scenarios/health_gathering.cfg")
# agentDQN_HealthGather.run_async_test("scenarios/health_gathering.cfg")
# agentDQN_HealthGather.replay_show("scenarios/health_gathering.cfg")

# Health Gather Supreme experiments
# agentDQN_HealthGatherSupreme = AgentsDQN.AgentDQN(model_name='NewDQN_HealthGatherSupreme6KNoDrop05_Real_1')
#agentDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")
# agentDQN_HealthGatherSupreme.run_final_test("scenarios/health_gathering_supreme.cfg")

# agentDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_HealthGatherSupremeNoDrop2')
#agentDuelDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")
# agentDuelDQN_HealthGatherSupreme.run_final_test("scenarios/health_gathering_supreme.cfg")

# agentDoubleDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_HealthGatherSupremeDrop6k')
#agentDoubleDuelDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")
# agentDoubleDuelDQN_HealthGatherSupreme.run_final_test("scenarios/health_gathering_supreme.cfg")

# agentA2C_HealthGather = AgentsA2C.A2C(model_name='A2C_healthGatherSupreme6kDrop05')
# agentA2C_HealthGather.run_final_test("scenarios/health_gathering_supreme.cfg")

# Health Gather Supreme experiments
# agentDQN_HealthGatherSupreme = AgentsDQN.AgentDQN(model_name='NewDQN_HealthGatherSupreme6KDrop05_Real_1')
#agentDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")
# agentDQN_HealthGatherSupreme.run_final_test("scenarios/health_gathering_supreme.cfg")

# agentDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_HealthGatherSupremeDrop6k')
#agentDuelDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")
# agentDuelDQN_HealthGatherSupreme.run_final_test("scenarios/health_gathering_supreme.cfg")

# agentDoubleDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_HealthGatherSupremeNoDrop2')
#agentDoubleDuelDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")
# agentDoubleDuelDQN_HealthGatherSupreme.run_final_test("scenarios/health_gathering_supreme.cfg")
# agentDoubleDuelDQN_HealthGatherSupreme.run_async_test("scenarios/health_gathering_supreme.cfg")
# agentDoubleDuelDQN_HealthGatherSupreme.replay_show("scenarios/health_gathering_supreme.cfg")

# agentA2C_HealthGather = AgentsA2C.A2C(model_name='A2C_healthGatherSupreme6k')
# agentA2C_HealthGather.run_final_test("scenarios/health_gathering_supreme.cfg")

"""Deadly Corridor experiments"""

# agentDQN_DeadlyCorridor = AgentsDQN.AgentDQN(model_name='NEW_WED_DQN_DeadlyCorridor_1')
# agentDQN_DeadlyCorridor.run_final_test("scenarios/deadly_corridor.cfg")
# agentDQN_DeadlyCorridor.run_async_test("scenarios/deadly_corridor.cfg")

# agentDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg", epoch_count=50)

# agentDuelDQN_DeadlyCorridor = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_DeadlyCorridor_final')
# agentDuelDQN_DeadlyCorridor.run_final_test("scenarios/deadly_corridor.cfg")
# agentDuelDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg", epoch_count=50)
# agentDuelDQN_DeadlyCorridor.run_async_test("scenarios/deadly_corridor.cfg")
# agentDuelDQN_DeadlyCorridor.replay_show("scenarios/deadly_corridor.cfg")

# agentDoubleDuelDQN_DeadlyCorridor = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_DeadlyCorridor_final')
# agentDoubleDuelDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg", epoch_count=50)
# agentDoubleDuelDQN_DeadlyCorridor.run_final_test("scenarios/deadly_corridor.cfg")

# agentA2C_DeadlyCorridor = AgentsA2C.A2C(model_name='A2C_Deadly_corridor_final')
# agentA2C_DeadlyCorridor.run_final_test("scenarios/deadly_corridor.cfg")
# agentA2C_DeadlyCorridor.exploration_decay = 0.9999
# agentA2C_DeadlyCorridor.exploration = 0.1
# agentA2C_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg", epoch_count=50, episodes_per_test=10)

# Tuning

# duelDDQN_Model_4 = Models.DuelNetworkConfigurable
# duelDDQN_Model_3 = Models.DuelNetworkConfigurable_3
# duelDDQN_Model_2 = Models.DuelNetworkConfigurable_2
# duelDDQN_Model_1 = Models.DuelNetworkConfigurable_1
#
# agentDuelDQN_Tuning_DefendTheLine_4_layers = AgentsDQN.AgentDuelDQN(model_name='Tuning_DuelDQN_DefendTheLine_4_Layers')
# agentDuelDQN_Tuning_DefendTheLine_4_layers.set_model(duelDDQN_Model_4)
# agentDuelDQN_Tuning_DefendTheLine_3_layers = AgentsDQN.AgentDuelDQN(model_name='Tuning_DuelDQN_DefendTheLine_3_Layers')
# agentDuelDQN_Tuning_DefendTheLine_3_layers.set_model(duelDDQN_Model_3)
# agentDuelDQN_Tuning_DefendTheLine_2_layers = AgentsDQN.AgentDuelDQN(model_name='Tuning_DuelDQN_DefendTheLine_2_Layers')
# agentDuelDQN_Tuning_DefendTheLine_2_layers.set_model(duelDDQN_Model_2)
# agentDuelDQN_Tuning_DefendTheLine_1_layers = AgentsDQN.AgentDuelDQN(model_name='Tuning_DuelDQN_DefendTheLine_1_Layers')
# agentDuelDQN_Tuning_DefendTheLine_1_layers.set_model(duelDDQN_Model_1)
#
# agentDuelDQN_Tuning_DefendTheLine_1_layers.start_training("scenarios/defend_the_line.cfg", episodes_per_test=10, episodes_per_epoch=100)
# agentDuelDQN_Tuning_DefendTheLine_2_layers.start_training("scenarios/defend_the_line.cfg")
# agentDuelDQN_Tuning_DefendTheLine_3_layers.start_training("scenarios/defend_the_line.cfg")
# agentDuelDQN_Tuning_DefendTheLine_4_layers.start_training("scenarios/defend_the_line.cfg")


#PPO
#agentA2CPPO_Basic = AgentsA2C.A2CPPO(model_name='A2CPPO_Basic', batch_size=256)
#agentA2CPPO_Basic.start_training("scenarios/basic.cfg", episodes_per_test=10)

#agentA2CPPO_HealthGather = AgentsA2C.A2CPPO(model_name='A2CPPO_HealthGather')
#agentA2CPPO_HealthGather.start_training("scenarios/health_gathering.cfg", episodes_per_test=1, epoch_count=50)


#agentA2CPPO_Multi1 = AgentsA2C.A2CPPO(model_name='A2CPPO_Multi1', batch_size=256)
#agentA2CPPO_Multi2 = AgentsA2C.A2CPPO(model_name='A2CPPO_Multi2', batch_size=256)

#DoomEnvironment.StartMultiplayerMatchTrain(agentA2CPPO_Multi1, agentA2CPPO_Multi2, config="scenarios/multi.cfg")



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

agent = AgentsA2C.A2CPPO(model_name=commands['model'], batch_size=512)
agent.start_training(commands['scenario'], episodes_per_test=100, fast_train=False)
