import os
import numpy as np
import matplotlib.pyplot as plt
from flatland_3_env import create_flatland_environment
from stable_baselines3 import PPO, DQN, A2C
from pettingzoo.utils import wrappers
from stable_baselines3.common.callbacks import BaseCallback

def run_experiment(env, model, num_episodes, render=False):
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

def main():

    # Constants
    # N_AGENTS = 5
    # N_EPISODES = 500
    # N_STEPS = 100

    # # Create the Flatland environment
    fl3_env = create_flatland_environment(
    width=30,
    height=30,
    num_agents=5,
    max_num_cities=4,
    grid_mode=False,
    max_rails_between_cities=2,
    max_rail_pairs_in_city=2,
    seed=39,
    observation_type="tree",
)

    pettingzoo_env = flatland_env.env(environment=fl3_env, use_renderer=False)

    # Define the list of experiments
    experiments = [
        {
            "question": 1,
            "speeds": [0.5, 1.0, 1.5, 2.0],
            "model": PPO,
        },
        {
            "question": 2,
            "algorithms": [PPO, DQN, A2C],
            "num_agents": 2,
        },
        {
            "question": 3,
            "num_agents_list": [1, 2, 3, 4],
            "model": PPO,
        },
        {
            "question": 5,
            "perception_levels": [1, 2, 3],
            "num_agents_list": [1, 2],
            "model": PPO,
        },
    ]

    for experiment in experiments:
        question = experiment["question"]
        if question == 1:
            # Question 1: Agent speed vs. performance
            # Implementation is the same as in the previous example
            # ...

        elif question == 2:
            # Question 2: Performance of different RL algorithms
            algorithms = experiment["algorithms"]
            num_agents = experiment["num_agents"]
            mean_rewards = []

            for algorithm in algorithms:
                env = create_flatland_env(num_agents=num_agents)  # Create environment with the given number of agents
                model = algorithm("MlpPolicy", env, verbose=1)
                model.learn(total_timesteps=10000)
                
                mean_reward = run_experiment(env, model, num_episodes=10)
                mean_rewards.append(mean_reward)
            
            algo_names = [algo.__name__ for algo in algorithms]
            plt.bar(algo_names, mean_rewards)
            plt.xlabel("RL Algorithm")
            plt.ylabel("Mean Reward")
            plt.title("Performance of Different RL Algorithms")
            plt.savefig("question2_results.png")
            plt.show()

        elif question == 3:
            # Question 3: Varying the number of agents
            num_agents_list = experiment["num_agents_list"]
            model_class = experiment["model"]
            mean_rewards = []

            for num_agents in num_agents_list:
                env = create_flatland_env(num_agents=num_agents)
                model = model_class("MlpPolicy", env, verbose=1)
                model.learn(total_timesteps=10000)

                mean_reward = run_experiment(env, model, num_episodes=10)
                mean_rewards.append(mean_reward)
            
            plt.plot(num_agents_list, mean_rewards)
            plt.xlabel("Number of Agents")
            plt.ylabel("Mean Reward")
            plt.title("Effect of Number of Agents on Performance")
            plt.savefig("question3_results.png")
            plt.show()




if __name__ == "__main__":
    main()