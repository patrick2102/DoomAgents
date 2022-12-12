import vizdoom

from src import DoomEnvironment, Tuning, Models
from src.Agents import Agent, AgentsDQN, AgentsA2C


"""Basic experiments"""
#agentDQN_Basic = AgentsDQN.AgentDQN(model_name='DQN_Basic')
#agentDQN_Basic.start_training("scenarios/basic.cfg")

#agentDuelDQN_Basic = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_Basic')
#agentDuelDQN_Basic.start_training("scenarios/basic.cfg")

#agentDoubleDuelDQN_Basic = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_Basic')
#agentDoubleDuelDQN_Basic.start_training("scenarios/basic.cfg")


"""Health Gather experiments"""

#agentDQN_HealthGather = AgentsDQN.AgentDQN(model_name='DQN_HealthGather')
#agentDQN_HealthGather.start_training("scenarios/health_gathering.cfg")

#agentDuelDQN_HealthGather = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_HealthGather')
#agentDuelDQN_HealthGather.start_training("scenarios/health_gathering.cfg")

#agentDoubleDuelDQN_HealthGather = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_HealthGather')
#agentDoubleDuelDQN_HealthGather.start_training("scenarios/health_gathering.cfg")


# Health Gather Supreme experiments
#agentDQN_HealthGatherSupreme = AgentsDQN.AgentDQN(model_name='DQN_HealthGatherSupreme')
#agentDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")

#agentDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_HealthGatherSupreme')
#agentDuelDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")

#agentDoubleDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_HealthGatherSupreme')
#agentDoubleDuelDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")



"""Deadly Corridor experiments"""


#agentDQN_DeadlyCorridor = AgentsDQN.AgentDQN(model_name='DQN_DeadlyCorridor')
#agentDQN_DeadlyCorridor.run_async_test("scenarios/deadly_corridor.cfg")

# agentDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg", epoch_count=50)

# agentDuelDQN_DeadlyCorridor = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_DeadlyCorridor')
# agentDuelDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg", epoch_count=50)
#
# agentDoubleDuelDQN_DeadlyCorridor = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_DeadlyCorridor')
# agentDoubleDuelDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg", epoch_count=50)

# Tuning

duelDDQN_Model_4 = Models.DuelNetworkConfigurable
duelDDQN_Model_3 = Models.DuelNetworkConfigurable_3
duelDDQN_Model_2 = Models.DuelNetworkConfigurable_2
duelDDQN_Model_1 = Models.DuelNetworkConfigurable_1

agentDuelDQN_Tuning_DefendTheLine_4_layers = AgentsDQN.AgentDuelDQN(model_name='Tuning_DuelDQN_DefendTheLine_4_Layers')
agentDuelDQN_Tuning_DefendTheLine_4_layers.set_model(duelDDQN_Model_4)
agentDuelDQN_Tuning_DefendTheLine_3_layers = AgentsDQN.AgentDuelDQN(model_name='Tuning_DuelDQN_DefendTheLine_3_Layers')
agentDuelDQN_Tuning_DefendTheLine_3_layers.set_model(duelDDQN_Model_3)
agentDuelDQN_Tuning_DefendTheLine_2_layers = AgentsDQN.AgentDuelDQN(model_name='Tuning_DuelDQN_DefendTheLine_2_Layers')
agentDuelDQN_Tuning_DefendTheLine_2_layers.set_model(duelDDQN_Model_2)
agentDuelDQN_Tuning_DefendTheLine_1_layers = AgentsDQN.AgentDuelDQN(model_name='Tuning_DuelDQN_DefendTheLine_1_Layers')
agentDuelDQN_Tuning_DefendTheLine_1_layers.set_model(duelDDQN_Model_1)

agentDuelDQN_Tuning_DefendTheLine_1_layers.start_training("scenarios/defend_the_line.cfg", episodes_per_test=10, episodes_per_epoch=100)
agentDuelDQN_Tuning_DefendTheLine_2_layers.start_training("scenarios/defend_the_line.cfg")
agentDuelDQN_Tuning_DefendTheLine_3_layers.start_training("scenarios/defend_the_line.cfg")
agentDuelDQN_Tuning_DefendTheLine_4_layers.start_training("scenarios/defend_the_line.cfg")


#Comments

#1 layer model:
"""
    Started out spinning and shooting the first enemies in the room, but then standing still until the time ran out.
    
    Afterwards it learned to go sideways and shoot to kill the first enemy. Never got much better after this.
"""

#2 layer model

"""
    Started spinning and shooting, then shot left enemy and moved sideways as above. Sometimes both enemies are shot.
"""