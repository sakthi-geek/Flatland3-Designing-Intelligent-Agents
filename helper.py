from stable_baselines3 import DQN, PPO, A2C

def get_model_class(agent_name):
    if agent_name == 'DQN':
        return DQN
    elif agent_name == 'PPO':
        return PPO
    elif agent_name == 'A2C':
        return A2C
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")