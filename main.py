import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from flatland.utils.rendertools import RenderTool
from pettingzoo.utils import wrappers
from pettingzoo.utils.to_parallel import from_parallel_wrapper
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from flatland_3_pettingzoo_env import Flatland3PettingZoo
from helper import get_model_class

# from stable_baselines3.common.results_plotter import load_results, ts2xy
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


#--- Parameters ------
TRAIN_FROM_CHECKPOINT = False

# Define experiments
experiments = {
    "experiment_1": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},
    
    "experiment_2": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},

    "experiment_3": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},

    "experiment_4": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},
    
    "experiment_5": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},
    
    "experiment_6": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},
    
    "experiment_7": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},
    
    "experiment_8": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},

    "experiment_9": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000},

    "experiment_10": {"env_config":{"seed": 39, "width": 25, "height": 25, "max_num_cities": 2, "grid_mode": False, 
                                   "max_rails_between_cities": 2, "max_rail_pairs_in_city": 2, "n_agents": 2, 
                                   "obs_type": "tree", "tree_max_depth": 2}, 
                    "agent": "DQN", "max_iterations": 10000, "checkpoint_interval": 1000}
}

#--- save the experiments to a json file ----------------
with open('experiments.json', 'w') as fp:
    json.dump(experiments, fp)


def run_experiment(exp_name, exp_params):
    # Initialize environment


    fl3_env = Flatland3PettingZoo(exp_params["env_config"])

    # Initialize the renderer
    env_renderer = RenderTool(fl3_env)
    # Reset the environment
    env_observe = fl3_env.reset()
    # Render the environment
    env_image = env_renderer.render_env(show=False, frames=False, show_observations=False, return_image=True)
    
    plt.imshow(env_image)
    #---save the environment image
    plt.savefig('env_image_1.png')
    #---show the environment imag
    plt.show()


    fl3_env = wrappers.OrderEnforcingWrapper(fl3_env)
    check_env(from_parallel_wrapper(DummyVecEnv([lambda: env])))

    
    # Initialize agent
    policy = 'MlpPolicy'
    if exp_params["agent"] == "DQN":
        model = DQN(policy, fl3_env, verbose=1)
    elif exp_params["agent"] == "PPO":
        model = PPO(policy, fl3_env, verbose=1)
    elif exp_params["agent"] == "A2C":
        model = A2C(policy, fl3_env, verbose=1)
    else:
        raise ValueError(f"Invalid agent: {exp_params['agent']}")

    # Create experiment directory
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save experiment configuration
    with open(os.path.join(exp_dir, "config.json"), "w") as config_file:
        json.dump(exp_params, config_file, indent=2)

    #--------------------- Add Callbacks ---------------------#
    # Create a checkpoint callback
    class CheckpointCallback(BaseCallback):

        def __init__(self, checkpoint_interval, experiment_dir):
            super().__init__()
            self.checkpoint_interval = checkpoint_interval
            self.experiment_dir = experiment_dir
            self.checkpoints = []

        def _on_step(self):
            if self.num_timesteps % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.experiment_dir, f"model_checkpoint_{self.num_timesteps}.zip")
                self.checkpoints.append(checkpoint_path)
                self.model.save(checkpoint_path)

            return True

    # Create a custom callback to collect metrics
    class MetricsCallback(BaseCallback):
        
        def __init__(self):
            super().__init__()
            self.rewards = []
            self.episode_lengths = []
            self.training_times = []
            self.completion_rates = []
            self.collision_rates = []
            self.episode_start_time = time.time()

        def _on_step(self):
            # Collect reward, episode length, and training time
            if self.locals.get("done"):
                self.rewards.append(self.locals["reward"])
                episode_length = self.num_timesteps - self.locals["self"].num_timesteps
                self.episode_lengths.append(episode_length)
                self.training_times.append(time.time() - self.episode_start_time)
                self.episode_start_time = time.time()

                # Collect completion rate and collision rate
                completion_rate = self.locals["env"].flatland_env.get_completion_rate()
                collision_rate = self.locals["env"].flatland_env.get_collision_rate()
                self.completion_rates.append(completion_rate)
                self.collision_rates.append(collision_rate)

            return True



    metrics_callback = MetricsCallback()

    if TRAIN_FROM_CHECKPOINT:
        checkpoint_callback = CheckpointCallback(checkpoint_interval=exp_params["checkpoint_interval"],
                                                experiment_dir=exp_dir)

        # Load the last available checkpoint if it exists
        last_checkpoint = None
        if os.path.exists(exp_dir):
            checkpoints = [filename for filename in os.listdir(exp_dir) if "model_checkpoint_" in filename]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=os.path.getctime)
                last_checkpoint = os.path.join(exp_dir, last_checkpoint)
                print(f"Resuming training from the last checkpoint: {last_checkpoint}")

        # Create and train the model
        model_class = get_model_class(exp_params["agent"]) # Get the model class


        if last_checkpoint:
            model = model_class.load(last_checkpoint, fl3_env, device=device)
            remaining_iterations = exp_params["max_iterations"] - int(last_checkpoint.split("_")[-1].split(".")[0])
        else:
            model = model_class(policy, fl3_env, verbose=1, device=device)
            remaining_iterations = exp_params["max_iterations"]

        model.learn(total_timesteps=remaining_iterations, callback=[checkpoint_callback, metrics_callback])
    #---------------------------------------------------------#
    # Train the agent
    model.learn(total_timesteps=exp_params["max_iterations"], callback=[checkpoint_callback, metrics_callback])

    # Save the trained model
    model.save(os.path.join(exp_dir, "model"))

    # Save metrics
    metrics = {
        "rewards": metrics_callback.rewards,
        "episode_lengths": metrics_callback.episode_lengths,
        "training_times": metrics_callback.training_times,
        "completion_rates": metrics_callback.completion_rates,
        "collision_rates": metrics_callback.collision_rates
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

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
    with open('experiments.json', 'r') as fp:
        experiments = json.load(fp)

    if args.exp:
        if args.exp in experiments:
            run_experiment(args.exp, experiments[args.exp])
        else:
            print(f"Experiment '{args.exp}' not found.")
    else:
        #--- run all the experiments -------------------------------
        for exp_name, exp_params in experiments.items():
            run_experiment(exp_name, exp_params)




