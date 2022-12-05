from src import DoomEnvironment, Tuning
from src.Agents import Agent, AgentsDQN, AgentsA2C


# Basic experiments

agentDQN_Basic = AgentsDQN.AgentDQN(model_name='DQN_Basic')
agentDQN_Basic.start_training("scenarios/basic.cfg")

"""
agentDuelDQN_Basic = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_Basic')
agentDuelDQN_Basic.start_training("scenarios/basic.cfg")

agentDoubleDuelDQN_Basic = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_Basic')
agentDoubleDuelDQN_Basic.start_training("scenarios/basic.cfg")
"""

# Health Gather experiments

"""
agentDQN_HealthGather = AgentsDQN.AgentDQN(model_name='DQN_HealthGather')
agentDQN_HealthGather.start_training("scenarios/health_gathering.cfg")

agentDuelDQN_HealthGather = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_HealthGather')
agentDuelDQN_HealthGather.start_training("scenarios/health_gathering.cfg")

agentDoubleDuelDQN_HealthGather = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_HealthGather')
agentDoubleDuelDQN_HealthGather.start_training("scenarios/health_gathering.cfg")
"""

# Health Gather Supreme experiments

"""
agentDQN_HealthGatherSupreme = AgentsDQN.AgentDQN(model_name='DQN_HealthGatherSupreme')
agentDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")

agentDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_HealthGatherSupreme')
agentDuelDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")

agentDoubleDuelDQN_HealthGatherSupreme = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_HealthGatherSupreme')
agentDoubleDuelDQN_HealthGatherSupreme.start_training("scenarios/health_gathering_supreme.cfg")
"""

# Deadly Corridor experiments

"""
agentDQN_DeadlyCorridor = AgentsDQN.AgentDQN(model_name='DQN_DeadlyCorridor')
agentDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg")

agentDuelDQN_DeadlyCorridor = AgentsDQN.AgentDuelDQN(model_name='DuelDQN_DeadlyCorridor')
agentDuelDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg")

agentDoubleDuelDQN_DeadlyCorridor = AgentsDQN.AgentDoubleDuelDQN(model_name='DoubleDuelDQN_DeadlyCorridor')
agentDoubleDuelDQN_DeadlyCorridor.start_training("scenarios/deadly_corridor.cfg")
"""