import argparse
import json
import os
import time
import pathlib

import matplotlib.pyplot as plt
import numpy as np
# from pettingzoo.utils import wrappers
# from stable_baselines3 import A2C, DQN, PPO
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecNormalize
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines3.common.monitor import Monitor
# from flatland.utils.rendertools import RenderTool
# import supersuit as ss
# from pettingzoo.utils.conversions import parallel_wrapper_fn
# from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
from pprint import pprint
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from ray.tune.registry import register_env
# import gymnasium as gym


# from flatland_3_pettingzoo_env import Flatland3PettingZoo
import other_helper


import torch
print(torch.cuda.is_available())





#--- Parameters ------
TRAIN_FROM_CHECKPOINT = False

# Define experiments
experiments = {
    "experiment_1": {"env_config": {"seed": 39, "width": 30, "height": 30,  "grid_mode": False, "obs_type": "tree", "tree_max_depth": 2,
                                   "max_num_cities": 3, "max_rails_between_cities": 3, "max_rail_pairs_in_city": 3, "num_agents": 3,
                                   "env_setting": "single", "num_envs": 1, "num_agents_per_env": 3,  # - can also be (single,1,2)
                                   "malfunction_params": {"malfunction_rate":1/1000, "min_duration":15,"max_duration":50}},
                    "agent": "DQN", "max_iterations": 2500, "model_prefix": "exp_1_model_", "model_save_path": "./experiment_1/models/",
                    "rl_library": "stable_baselines3", "device": "cuda", "policy": "MlpPolicy",  # - can also be "rllib"
                    "checkpoint_save_freq": 100, "checkpoint_prefix": "exp_1_checkpoint_", "checkpoint_save_path": "./experiment_1/checkpoints/",
                    "n_eval_episodes": 25, "eval_freq": 100, "eval_save_path": "./experiment_1/eval/", "eval_log_name": "eval_results.txt"
                     }
}
        ###
#--- save the experiments to a json file ----------------
# with open('flatland3_MARL_experiments.json', 'w') as fp:
#     json.dump(experiments, fp)
# print("Experiments saved to {} file".format('flatland3_MARL_experiments.json'))

#-----------------------------------------------------------------------------------------------------
# # --------------------- Add Callbacks ---------------------#
# # Create a checkpoint callback
# class CheckpointCallback(BaseCallback):
#     def __init__(self, checkpoint_save_freq, experiment_dir):
#         super().__init__()
#         self.checkpoint_save_freq = checkpoint_save_freq
#         self.experiment_dir = experiment_dir
#         self.checkpoints = []
#
#     def _on_step(self):
#         if self.num_timesteps % self.checkpoint_save_freq == 0:
#             checkpoint_path = os.path.join(self.experiment_dir, f"model_checkpoint_{self.num_timesteps}.zip")
#             print(f"{self.num_timesteps} - Saving checkpoint to {checkpoint_path}")
#             self.checkpoints.append(checkpoint_path)
#             self.model.save(checkpoint_path)
#
#         return True
#
# # Create a custom callback to collect metrics
# class MetricsCallback(BaseCallback):
#
#     def __init__(self):
#         super().__init__()
#         self.rewards = []
#         self.episode_lengths = []
#         self.training_times = []
#         self.completion_rates = []
#         self.collision_rates = []
#         self.episode_start_time = time.time()
#
#     def _on_step(self):
#         # Collect reward, episode length, and training time
#         if self.locals.get("done"):
#             self.rewards.append(self.locals["reward"])
#             episode_length = self.num_timesteps - self.locals["self"].num_timesteps
#             self.episode_lengths.append(episode_length)
#             self.training_times.append(time.time() - self.episode_start_time)
#             self.episode_start_time = time.time()
#
#             # Collect completion rate and collision rate
#             completion_rate = self.locals["env"].flatland_env.get_completion_rate()
#             collision_rate = self.locals["env"].flatland_env.get_collision_rate()
#             self.completion_rates.append(completion_rate)
#             self.collision_rates.append(collision_rate)
#
#         return True

def run_experiment(exp_name, exp_params):

    # Create experiment directory
    exp_dir = "experiments"
    pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Save experiment configuration to a json file
    exp_config_path = os.path.join(exp_dir, "{}_config.json".format(exp_name))
    with open(exp_config_path, "w") as config_file:
        json.dump(exp_params, config_file, indent=2)
    print("Experiment configuration saved to {} file".format(exp_config_path))

    env_config = exp_params["env_config"]

    # save the env_config to a json file
    env_config_path = os.path.join(exp_dir, "{}_env_config.json".format(exp_name))
    with open(env_config_path, "w") as config_file:
        json.dump(env_config, config_file, indent=2)
    print("Environment configuration saved to {} file".format(env_config_path))

    # Initialize environment
    env_setting = env_config["env_setting"]
    num_envs = env_config["num_envs"]
    seed = env_config["seed"]
    obs_params = exp_params["obs_params"]

    ##------ flatland3 environment ----------------
    fl3_env = other_helper.create_flatland3_env(env_config, obs_params)
    # - render the env --------
    other_helper.render_env(fl3_env)
    print("fl3 Environment initialized")
    sdsd
    obs = fl3_env._get_observations()
    print(len(obs[0]))
    pprint(obs[0])

    n_features_per_node = fl3_env.obs_builder.observation_dim
    print(n_features_per_node)
    n_nodes = sum([np.power(4, i) for i in range(env_config["tree_max_depth"] + 1)])
    print(n_nodes)
    state_size = n_features_per_node * n_nodes
    print(state_size)

    print(fl3_env.observation_spaces)
    print(fl3_env.action_spaces)


    # # -- prepare the env for stable baselines3

    fl3_venv = make_vec_env(lambda: helper.create_flatland3_env(env_config),
        n_envs=1, vec_env_cls=DummyVecEnv)
    fl3_venv = VecNormalize(fl3_venv)

    # metadata = {"is_parallelizable": True}
    # fl3_env.metadata = metadata
    # fl3_env = aec_to_parallel(fl3_env)
    # fl3_env = ss.stable_baselines3_vec_env_v0(gym.make(fl3_env), num_envs)
    # fl3_env = ss.concat_vec_envs_v1(fl3_env, num_envs, num_cpus=1, base_class="stable_baselines3")

    #------ flatland3 pettingzoo environment ----------------
    # fl3_pz_env = Flatland3PettingZoo(env_config)
    #
    # print("fl3 pz Environment initialized")
    # print(fl3_pz_env.action_spaces)
    # print(fl3_pz_env.observation_spaces)
    #
    # #-- prepare the env for stable baselines3
    # metadata = {"is_parallelizable": True}
    # fl3_pz_env.metadata = metadata
    # fl3_pz_env = aec_to_parallel(fl3_pz_env)
    # fl3_pz_env = ss.pettingzoo_env_to_vec_env_v1(fl3_pz_env)
    # fl3_pz_env = ss.concat_vec_envs_v1(fl3_pz_env, num_envs, num_cpus=1, base_class="stable_baselines3")

    # Prepare the environment for RLlib
    # def env_creator(env_config):  ##--- still in progress ------
    #     env = helper.create_flatland3_env(env_config)
    #     env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,))
    #     return env
    #
    # env_name = "flatland3_pettyzoov0"
    # register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(env_config)))
    # fl3_env = ParallelPettingZooEnv(env_creator(env_config))

    ## Check environment
    # check_env(fl3_env, warn=True, skip_render_check=False)

    # Initialize agent
    policy = exp_params["policy"]
    device = exp_params["device"]    # device = "cuda" if torch.cuda.is_available() else "cpu"

    if exp_params["agent"] == "DQN":
        model = DQN(policy, fl3_venv, verbose=1, seed=seed, device=device)
    elif exp_params["agent"] == "PPO":
        model = PPO(policy, fl3_env, verbose=1, seed=seed, device=device)
    elif exp_params["agent"] == "A2C":
        model = A2C(policy, fl3_env, verbose=1, seed=seed, device=device)
    else:
        raise ValueError(f"Invalid agent: {exp_params['agent']}")
    print("Agent initialized")

    # Create callbacks
    metrics_callback = MetricsCallback()
    checkpoint_callback = CheckpointCallback(checkpoint_save_freq=exp_params["checkpoint_save_freq"],
                                             experiment_dir=exp_dir)

    # Train the agent from last checkpoint if it exists
    if TRAIN_FROM_CHECKPOINT:
        print("Training from last checkpoint")
        # Load the last available checkpoint if it exists
        last_checkpoint = None
        if os.path.exists(exp_dir):
            checkpoints = [filename for filename in os.listdir(exp_dir) if "model_checkpoint_" in filename]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=os.path.getctime)
                last_checkpoint = os.path.join(exp_dir, last_checkpoint)
                print(f"Resuming training from the last checkpoint: {last_checkpoint}")

        # Create and train the model
        model_class = helper.get_model_class(exp_params["agent"])  # Get the model class

        if last_checkpoint:
            model = model_class.load(last_checkpoint, fl3_env, verbose=1, seed=seed, device=device)
            remaining_iterations = exp_params["max_iterations"] - int(last_checkpoint.split("_")[-1].split(".")[0])
        else:
            model = model_class(policy, fl3_env, verbose=1, seed=seed, device=device)
            remaining_iterations = exp_params["max_iterations"]

        model.learn(total_timesteps=remaining_iterations, callback=[checkpoint_callback, metrics_callback])
    else:
        # Train the agent from scratch
        model.learn(total_timesteps=exp_params["max_iterations"], callback=[checkpoint_callback, metrics_callback])
    print("Training completed")

    # Save the trained model
    model_save_path = env_params["model_save_path"]
    print(f"Saving the trained model to {model_save_path}")
    model.save(model_save_path)

    # Evaluate the model
    mean_reward, std_reward = stable_baselines3.common.evaluation.evaluate_policy(
        model,
        evaluation_env,
        n_eval_episodes=training_params["n_eval_episodes"],
        deterministic=True,
        return_episode_rewards=True,
    )

    # Log the evaluation results
    with open(training_params["eval_save_path"] + training_params["eval_log_name"], "w") as f:
        f.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save metrics
    metrics = {
        "rewards": metrics_callback.rewards,
        "episode_lengths": metrics_callback.episode_lengths,
        "training_times": metrics_callback.training_times,
        "completion_rates": metrics_callback.completion_rates,
        "collision_rates": metrics_callback.collision_rates
    }
    with open(os.path.join(exp_dir, "{}_metrics.json".format(exp_name)), "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print("Metrics saved successfully")

    # Plot the results
    results_plotter.plot_results(
        [training_params["checkpoint_save_path"]],
        training_params["max_iterations"],
        results_plotter.X_TIMESTEPS,
        "MARL Experiment: " + training_params["model_prefix"],
    )
    plt.savefig(training_params["eval_save_path"] + "training_plot.png")

    # Plot reward, episode length, training time, completion rate, and collision rate
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    axs[0, 0].plot(callback.rewards)
    axs[0, 0].set_title("Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 1].plot(callback.episode_lengths)
    axs[0, 1].set_title("Episode Length")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Length")
    axs[1, 0].plot(callback.training_times)
    axs[1, 0].set_title("Training Time")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Time (s)")
    axs[1, 1].plot(callback.completion_rates)
    axs[1, 1].set_title("Completion Rate")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Completion Rate")
    axs[2, 0].plot(callback.collision_rates)
    axs[2, 0].set_title("Collision Rate")
    axs[2, 0].set_xlabel("Episode")
    axs[2, 0].set_ylabel("Collision Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatland3 PettingZoo Experiments")
    parser.add_argument("--exp", type=str, default=None, help="Experiment name to run (default: None, run all experiments)")

    args = parser.parse_args()

    #--- load the experiments from a json file --------------
    with open('flatland3_MARL_experiments.json', 'r') as fp:
        experiments = json.load(fp)
    print("Loaded {} experiments from {} file".format(len(experiments), 'flatland3_MARL_experiments.json'))

    if args.exp:
        if args.exp in experiments:
            print("Running experiment - {}".format(args.exp))
            run_experiment(args.exp, experiments[args.exp])
        else:
            print(f"Experiment '{args.exp}' not found.")
    else:
        #--- run all the experiments -------------------------------
        print("Running all experiments")
        for exp_name, exp_params in experiments.items():
            run_experiment(exp_name, exp_params)




