from datetime import datetime
import json
import csv
import os
import pathlib
import sys
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint
import psutil
from types import SimpleNamespace
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


from flatland.envs.observations import (GlobalObsForRailEnv, LocalObsForRailEnv, TreeObsForRailEnv)
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv   ###
from flatland.utils.rendertools import RenderTool
from flatland.envs.step_utils.states import TrainState
from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatland.envs.agent_chains import MotionCheck


# from custom_rail_env import RailEnv
from custom_observations import CombinedLocalGlobalObs
from reinforcement_learning.dddqn_policy import DDDQNPolicy
from utils import Timer
from utils import normalize_observation

base_dir = Path(__file__).resolve().parent
print(base_dir)
sys.path.append(str(base_dir))

try:
    import wandb
    #--- get the wandb api key from the file ----
    with open("wandb_api_key.txt", "r") as f:
        api_key = f.readline()
    wandb.login(key=api_key)
    wandb.init(sync_tensorboard=True)
except ImportError:
    print("Install wandb to log to Weights & Biases")


def convert_to_supported_types(value):
    if isinstance(value, (int, float, str, bool, torch.Tensor)):
        return value
    elif isinstance(value, list):
        return str(value)  # Convert list to string
    elif isinstance(value, np.ndarray):
        return torch.tensor(value)  # Convert numpy array to torch.Tensor
    else:
        return str(value)  # Convert any other type to string

def initial_prep_for_experiment(exp_name, exp_params):
    # Create experiment directory
    exp_dir = "experiments/{}".format(exp_name)
    pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=True)
    print("Experiment directory created at {}".format(exp_dir))

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


def create_flatland3_env(env_config, obs_params):

    _seed = env_config.get("seed")
    n_agents = env_config.get("n_agents")
    width = env_config.get("width")
    height = env_config.get("height")
    max_num_cities = env_config.get("max_num_cities")
    grid_mode = env_config.get("grid_mode")
    max_rails_between_cities = env_config.get("max_rails_between_cities")
    max_rail_pairs_in_city = env_config.get("max_rail_pairs_in_city")
    agents = [f'train_{i}' for i in range(env_config.get("n_agents"))]
    obs_type = env_config.get("obs_type")
    predictor = env_config.get("predictor")
    malfunction_params = env_config.get("malfunction_params", {})
    malfunction_rate = malfunction_params.get("malfunction_rate")
    min_duration = malfunction_params.get("min_duration")
    max_duration = malfunction_params.get("max_duration")

    # Observation parameters
    max_tree_depth = obs_params.get("max_tree_depth", 2)  # 2 default
    observation_radius = obs_params.get("observation_radius", 10)  # 10 default
    max_path_depth = obs_params.get("max_path_depth", 20)   # 20 default

    ## Initialize Flatland environment
    rail_generator = sparse_rail_generator(
        max_num_cities=max_num_cities,
        grid_mode=grid_mode,
        max_rails_between_cities=max_rails_between_cities,
        max_rail_pairs_in_city=max_rail_pairs_in_city,
    )
    line_generator = sparse_line_generator()
    if obs_type == 'tree':
        if predictor == "shortest_path":
            obs_builder = TreeObsForRailEnv(max_depth=max_tree_depth, predictor=ShortestPathPredictorForRailEnv(max_path_depth))
        else:
            obs_builder = TreeObsForRailEnv(max_depth=max_tree_depth)
    elif obs_type == 'local':
        obs_builder = LocalObsForRailEnv(view_width=width//3, view_height=width//3, center=0)
    elif obs_type == 'global':
        obs_builder = GlobalObsForRailEnv()
    elif obs_type == "combined":
        if predictor == "shortest_path":
            obs_builder = CombinedLocalGlobalObs(tree_depth=max_tree_depth, predictor=ShortestPathPredictorForRailEnv(max_path_depth), local_radius=width//3)
        else:
            obs_builder = CombinedLocalGlobalObs(tree_depth=max_tree_depth, local_radius=width//3)
    else:
        raise ValueError(f"Invalid observation type: {obs_type}")

    if malfunction_params:
        stochastic_data = MalfunctionParameters(
            malfunction_rate=malfunction_rate,  #1 / 10000,  # Rate of malfunction occurence
            min_duration=min_duration,          #15,  # Minimal duration of malfunction
            max_duration=max_duration)           #50  # Max duration of malfunction
        malfunction_generator = ParamMalfunctionGen(stochastic_data)
    else:
        malfunction_generator = None

    fl3_env = RailEnv(
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        malfunction_generator=malfunction_generator,
        number_of_agents=n_agents,
        obs_builder_object=obs_builder,
        random_seed=_seed
    )

    return fl3_env


def train_agent(exp_name, exp_params, train_env, eval_env, obs_params):

    # Unique ID for this training
    now = datetime.now()
    training_id = exp_name + '_' + now.strftime('%y%m%d%H%M%S')

    # Environment parameters
    train_env_config = exp_params.get("env_config")
    n_agents = exp_params.get("env_config")["n_agents"]
    width = exp_params.get("env_config")["width"]
    height = exp_params.get("env_config")["height"]
    max_num_cities = exp_params.get("env_config")["max_num_cities"]
    max_rails_between_cities = exp_params.get("env_config")["max_rails_between_cities"]
    max_rail_pairs_in_city = exp_params.get("env_config")["max_rail_pairs_in_city"]
    seed = exp_params.get("env_config")["seed"]

    # Observation parameters
    max_tree_depth = obs_params.get("max_tree_depth", 2)            # 2 default
    observation_radius = obs_params.get("observation_radius", 10)                   # 10 default
    max_path_depth = obs_params.get("max_path_depth", 20)   # 20 default

    # Training parameters
    eps_start = exp_params.get("epsilon_start", 1.0)                                # 1.0 default
    eps_end = exp_params.get("epsilon_end", 0.01)                                   # 0.01 default
    eps_decay = exp_params.get("epsilon_decay", 0.99)                               # 0.99 default
    n_episodes = exp_params.get("n_episodes", 2500)                                # 2500 default
    checkpoint_save_freq = exp_params.get("checkpoint_save_freq", 100)              # 100 default
    n_eval_episodes = exp_params.get("n_evaluation_episodes", 25)                   # 25 default
    restore_replay_buffer = exp_params.get("restore_replay_buffer", False)          # False default
    save_replay_buffer = exp_params.get("save_replay_buffer", False)                # False default
    print("Loaded parameters:\n")

    # Set the seeds
    random.seed(seed)

    #--------- metrics initialization ------------
    metrics_dir = "metrics/"
    pathlib.Path(metrics_dir).mkdir(parents=True, exist_ok=True)

    # Initialize metrics dictionary
    metrics_dict = init_metrics()

    # Create a metrics log file for training and eval logs
    metrics_csv_file = metrics_dir + '{}_metrics.csv'.format(training_id)

    # Write headers to log files
    write_logs_to_csv(metrics_csv_file, ['episode', 'score', 'smoothed_score', 'completion', 'smoothed_completion', 
            'travel_time', 'average_travel_time', 'normalized_deadlocks', 'eps_start', 'action_probs-[B, L, F, R, S]',
            'n_eval_episodes', 'eval_score', 'eval_completion', 'eval_travel_time', 'eval_deadlocks'])

    print("Metrics initialized")



    # Setup renderer
    if exp_params.get("render"):
        env_renderer = RenderTool(train_env, gl="PGL")
        print("Environment Renderer initialized")

    #-------- Calculate state_size for the observation space ------------
    if exp_params.get("env_config").get("obs_type") == "tree":
        # Calculate the state size given the depth of the tree observation and the number of features
        n_features_per_node = train_env.obs_builder.observation_dim
        n_nodes = sum([np.power(4, i) for i in range(max_tree_depth + 1)])
        state_size = n_features_per_node * n_nodes
    elif exp_params.get("env_config").get("obs_type") == "global":
        state_size = height * width * (16 + 5 + 2)
    elif exp_params.get("env_config").get("obs_type") == "local":
        view_width = width // 3
        view_height = height // 3
        state_size = view_height * (2 * view_width + 1) * (16 + 2 + 2 + 4)
    elif exp_params.get("env_config").get("obs_type") == "combined":
        n_features_per_node = train_env.obs_builder.observation_dim
        n_nodes = sum([np.power(4, i) for i in range(max_tree_depth + 1)])
        tree_state_size = n_features_per_node * n_nodes
        local_area_size = (2 * observation_radius + 1) * (2 * observation_radius + 1) * (16 + 5 + 2)
        state_size = tree_state_size + local_area_size
    else:
        raise ValueError("observation type not recognized - state size cannot be calculated")

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_eval_normalized_score = -1.0
    smoothed_completion = 0.0
    smoothed_eval_completion = 0.0

    # Double Dueling DQN policy
    policy = DDDQNPolicy(state_size, action_size, exp_params)
    print("Policy loaded")

    # Loads existing replay buffer
    if restore_replay_buffer:
        try:
            policy.load_replay_buffer(restore_replay_buffer)
            policy.test()
        except RuntimeError as e:
            print("\n Could't load replay buffer, were the experiences generated using the same tree depth?")
            print(e)
            exit(1)

    print("\n Replay buffer status: {}/{} experiences".format(len(policy.memory.memory), exp_params.get("buffer_size")))

    hdd = psutil.disk_usage('/')
    if save_replay_buffer and (hdd.free / (2 ** 30)) < 500.0:
        print(
            "Careful! Saving replay buffers will quickly consume a lot of disk space. You have {:.2f}gb left.".format(
                hdd.free / (2 ** 30)))

    # # Convert exp_params dictionary to an object
    # exp_params_obj = SimpleNamespace(**exp_params)
    # train_env_config_obj = SimpleNamespace(**train_env_config)
    # obs_params_obj = SimpleNamespace(**obs_params)

    # Convert all values to supported types
    exp_params = {k: convert_to_supported_types(v) for k, v in exp_params.items()}
    train_env_config = {k: convert_to_supported_types(v) for k, v in train_env_config.items()}
    obs_params_obj = {k: convert_to_supported_types(v) for k, v in obs_params.items()}

    # TensorBoard writer
    writer = SummaryWriter()
    print("Tensorboard writer initialized")
    writer.add_hparams(exp_params, {})
    writer.add_hparams(train_env_config, {})
    writer.add_hparams(obs_params_obj, {})

    print("Starting training...\n")
    training_timer = Timer()
    training_timer.start()

    print("\n Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}'.\n".format(
            n_agents,
            width, height,
            n_episodes,
            n_eval_episodes,
            checkpoint_save_freq,
            training_id
        ))

    total_nb_steps = 0
    for episode_idx in range(n_episodes):
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)
        reset_timer.end()

        # Init these values after reset()
        max_steps = train_env._max_episode_steps
        action_count = [0] * action_size
        action_dict = dict()
        agent_obs = [None] * n_agents
        agent_prev_obs = [None] * n_agents
        agent_prev_action = [2] * n_agents
        update_values = [False] * n_agents

        if exp_params.get("render"):
            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []
        direct_deadlock = 0

        # Build initial agent-specific observations
        for agent in train_env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], max_tree_depth,
                                                         observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for step in range(max_steps):
            inference_timer.start()
            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent_obs[agent], eps=eps_start)

                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            step_timer.end()

            # Render an episode at some interval
            if exp_params.get("render") and episode_idx % checkpoint_save_freq == 0:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    learn_timer.start()
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent],
                                done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if next_obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(next_obs[agent], max_tree_depth,
                                                             observation_radius=observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent]

            # ------------ deadlock detection ------------#
            motion_check = MotionCheck()
            # Add agents to the MotionCheck instance
            for agent_idx, agent in enumerate(train_env.agents):
                position = agent.position
                next_position = tuple(agent.target)
                motion_check.addAgent(agent_idx, position, next_position)

            # Find conflicts using the find_conflicts() function
            motion_check.find_conflicts()

            # Calculate the number of direct deadlocks in the current step -  high risk of collision
            direct_deadlocks_in_step = sum(
                [1 for node, attr in motion_check.G.nodes.items() if attr.get('color') == 'purple'])

            direct_deadlock += direct_deadlocks_in_step

            nb_steps = step

            if done['__all__']:
                break

        total_nb_steps += nb_steps
        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum([agent.state == TrainState.DONE for agent in train_env.agents])
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())
        normalized_deadlocks = direct_deadlock / (max_steps * train_env.get_num_agents())
        mean_nb_steps = total_nb_steps / (episode_idx + 1)

        # if no actions were ever taken possibly due to malfunction and so
        # - `actions_taken` is empty [],
        # - `np.sum(action_count)` is 0
        # Set action probs to count
        if (np.sum(action_count) > 0):
            action_probs = action_count / np.sum(action_count)
        else:
            action_probs = action_count
        action_count = [1] * action_size

        # Set actions_taken to a list with single item 0
        if not actions_taken:
            actions_taken = [0]

        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        # Print logs
        episode_idx += 1 # Increment episode counter for checkpoint saving
        if episode_idx % checkpoint_save_freq == 0:
            torch.save(policy.qnetwork_local, './checkpoints/' + training_id + '-' + str(episode_idx) + '.pth')

            if save_replay_buffer:
                policy.save_replay_buffer('./replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')

            if exp_params.get("render"):
                env_renderer.close_window()

            # Log metrics
        print(
            '\r Episode {}'
            '\t  Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t  Done: {:.2f}%'
            ' Avg: {:.2f}%'
            '\t Travel_Time {:.2f}'
            '\t Avg: {:.2f}'
            '\t  Deadlock: {:.2f}'
            '\t  Epsilon: {:.3f} '
            '\t  Action Probs: {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                nb_steps,
                mean_nb_steps,
                normalized_deadlocks,
                eps_start,
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluate policy and log results at some interval
        if episode_idx % checkpoint_save_freq == 0 and n_eval_episodes > 0:
            scores, completions, nb_steps_eval, deadlocks = eval_policy(eval_env, policy, exp_params, obs_params)
            
            #-------- Log training and evaluation metrics --------
            # Update metrics
            update_train_metrics(metrics_dict, episode_idx, normalized_score, smoothed_normalized_score,
                                    completion, smoothed_completion, nb_steps, mean_nb_steps, normalized_deadlocks, 
                                    eps_start, action_probs.tolist())

            update_eval_metrics(metrics_dict, episode_idx, n_eval_episodes, np.mean(scores),
                                    np.mean(completions), np.mean(nb_steps_eval), np.mean(deadlocks))
            
            # Save metrics log to the CSV file
            write_logs_to_csv(metrics_csv_file, [episode_idx, normalized_score, smoothed_normalized_score, completion,
                    smoothed_completion, nb_steps, mean_nb_steps, normalized_deadlocks, eps_start, str(action_probs),
                    n_eval_episodes, np.mean(scores), np.mean(completions), np.mean(nb_steps_eval), np.mean(deadlocks)])

            print("Metrics logged.")

            writer.add_scalar("evaluation/scores_min", np.min(scores), episode_idx)
            writer.add_scalar("evaluation/scores_max", np.max(scores), episode_idx)
            writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode_idx)
            writer.add_scalar("evaluation/scores_std", np.std(scores), episode_idx)
            writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
            writer.add_scalar("evaluation/completions_min", np.min(completions), episode_idx)
            writer.add_scalar("evaluation/completions_max", np.max(completions), episode_idx)
            writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode_idx)
            writer.add_scalar("evaluation/completions_std", np.std(completions), episode_idx)
            writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
            writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode_idx)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)

            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (
                        1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode_idx)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode_idx)

        # Save logs to tensorboard
        writer.add_scalar("training/score", normalized_score, episode_idx)
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
        writer.add_scalar("training/completion", np.mean(completion), episode_idx)
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode_idx)
        writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
        writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode_idx)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
        writer.add_scalar("training/epsilon", eps_start, episode_idx)
        writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
        writer.add_scalar("training/loss", policy.loss, episode_idx)
        writer.add_scalar("timer/reset", reset_timer.get(), episode_idx)
        writer.add_scalar("timer/step", step_timer.get(), episode_idx)
        writer.add_scalar("timer/learn", learn_timer.get(), episode_idx)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode_idx)
        writer.add_scalar("timer/total", training_timer.get_current(), episode_idx)

    # Save metrics dict to the json file
    metrics_json_file = metrics_dir + '/{}_metrics.json'.format(training_id)
    save_metrics_to_json(metrics_dict, metrics_json_file)


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(env, policy, exp_params, obs_params):
    n_eval_episodes = exp_params.get("n_eval_episodes")
    tree_depth = obs_params.get("max_tree_depth")
    observation_radius = obs_params.get("observation_radius")

    scores = []
    completions = []
    nb_steps = []
    direct_deadlocks = []

    for episode_idx in range(n_eval_episodes):

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        max_steps = env._max_episode_steps
        action_dict = dict()
        agent_obs = [None] * env.get_num_agents()
        direct_deadlock = 0


        score = 0.0

        final_step = 0

        for step in range(max_steps):
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(obs[agent], tree_depth=tree_depth,
                                                             observation_radius=observation_radius)

                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            #------------ deadlock detection ------------#
            motion_check = MotionCheck()
            # Add agents to the MotionCheck instance
            for agent_idx, agent in enumerate(env.agents):
                position = agent.position
                next_position = tuple(agent.target)
                motion_check.addAgent(agent_idx, position, next_position)

            # Find conflicts using the find_conflicts() function
            motion_check.find_conflicts()

            # Calculate the number of direct deadlocks in the current step -  high risk of collision
            direct_deadlocks_in_step = sum(
                [1 for node, attr in motion_check.G.nodes.items() if attr.get('color') == 'purple'])

            direct_deadlock += direct_deadlocks_in_step

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum([agent.state == TrainState.DONE for agent in env.agents])
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        normalized_direct_deadlocks = direct_deadlock / (max_steps * env.get_num_agents())
        direct_deadlocks.append(normalized_direct_deadlocks)

        nb_steps.append(final_step)
        
    print("\t Eval: score {:.3f} done {:.1f}% avg_travel_time {:.2f} avg_deadlocks {:.2f}".format(np.mean(scores),
                                                                    np.mean(completions) * 100.0,
                                                                    np.mean(nb_steps),
                                                                   np.mean(direct_deadlocks)))

    return scores, completions, nb_steps, direct_deadlocks


def write_logs_to_csv(filename, logs):
    with open(filename, 'a', newline='') as csvfile:
        log_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(logs)


def init_metrics():
    metrics = {
        "training": {
            "episode": [],
            "score": [],
            "smoothed_score": [],
            "completion": [],
            "smoothed_completion": [],
            "travel_time": [],
            "average_travel_time": [],
            "normalized_deadlocks": [],
            "eps_start": [],
            "action_probs-[B, L, F, R, S]": []
        },
        "evaluation": {
            "episode": [],
            "n_eval_episodes": [],
            "eval_score": [],
            "eval_completion": [],
            "eval_travel_time": [],
            "eval_deadlocks": []
        }
    }
    return metrics


def update_train_metrics(metrics, episode, score, smoothed_score, completion, smoothed_completion, travel_time,
                         average_travel_time, normalized_deadlocks, eps_start, action_probs):
    metrics["training"]["episode"].append(episode)
    metrics["training"]["score"].append(score)
    metrics["training"]["smoothed_score"].append(smoothed_score)
    metrics["training"]["completion"].append(completion)
    metrics["training"]["smoothed_completion"].append(smoothed_completion)
    metrics["training"]["travel_time"].append(travel_time)
    metrics["training"]["average_travel_time"].append(average_travel_time)
    metrics["training"]["normalized_deadlocks"].append(normalized_deadlocks)
    metrics["training"]["eps_start"].append(eps_start)
    metrics["training"]["action_probs-[B, L, F, R, S]"].append(action_probs)


def update_eval_metrics(metrics, episode, n_eval_episodes, mean_scores, mean_completions, mean_travel_time,
                        normalized_deadlocks):
    metrics["evaluation"]["episode"].append(episode)
    metrics["evaluation"]["n_eval_episodes"].append(n_eval_episodes)
    metrics["evaluation"]["eval_score"].append(mean_scores)
    metrics["evaluation"]["eval_completion"].append(mean_completions)
    metrics["evaluation"]["eval_travel_time"].append(mean_travel_time)
    metrics["evaluation"]["eval_deadlocks"].append(normalized_deadlocks)


def save_metrics_to_json(metrics, filepath):
    with open(filepath, 'w') as file:
        json.dump(metrics, file, indent=4)


def load_metrics_from_json(filepath):
    with open(filepath, 'r') as file:
        metrics = json.load(file)
    return metrics


def convert_csv_to_json(csv_file_path, json_file_path, type='custom'):
    df = pd.read_csv(csv_file_path)
    if type == 'custom':
        # Convert each column into a dictionary of key-value pairs
        dict = {"training": {}, "evaluation": {}}
        for column in df.columns:
            if "eval" not in column:
                dict["training"][column] = df[column].tolist()
                if "episode" in column:
                    dict["evaluation"][column] = df[column].tolist()
            else:
                dict["evaluation"][column] = df[column].tolist()

        # Convert the dictionary into a json file
        with open(json_file_path, 'w') as json_file:
            json.dump(dict, json_file, indent=4)
    else:
        df.to_json(json_file_path, orient='records', indent=4)


def plot_metrics(metrics, metric_name):
    plt.plot(metrics["training"][metric_name], label="Training")
    plt.plot(metrics["evaluation"][metric_name], label="Evaluation")
    plt.xlabel("Episodes")
    plt.ylabel(metric_name.capitalize().replace("_", " "))
    plt.legend()
    plt.show()


def combine_multiple_experiments_csv(csv_files):
    combined_df = pd.DataFrame()
    experiment_key_list = []

    for filename in csv_files:
        df = pd.read_csv(filename)

        # ---- get experiment name from the file name ----
        experiment_name = "_".join(filename.split(os.sep)[1].split("_")[0:2])
        # print(experiment_name)
        # - add experiment name as a column in the dataframe
        df['experiment_name'] = experiment_name

        # ----get experiment key from the file name ----
        experiment_key = filename.split("_")[1]
        # print(experiment_key)
        # --- add experiment key to the list
        experiment_key_list.append(experiment_key)

        combined_df = pd.concat([combined_df, df])

    new_combined_csv_path = 'metrics/all_experiments_{}_{}_metrics.csv'.format(experiment_key_list,
                                                                           datetime.now().strftime('%y%m%d%H%M%S'))
    combined_df.to_csv(new_combined_csv_path, index=False)

    return combined_df


def combine_multiple_experiments_json(json_files):
    multi_experiment_metrics_json = {}
    experiment_key_list = []

    for filename in json_files:
        with open(filename) as f:
            experiment_json = json.load(f)

        # ---- get experiment name from the file name ----
        experiment_name = "_".join(filename.split(os.sep)[1].split("_")[0:2])
        # print(experiment_name)
        # ----get experiment key from the file name ----
        experiment_key = filename.split("_")[1]
        # print(experiment_key)
        # --- add experiment key to the list
        experiment_key_list.append(experiment_key)
        multi_experiment_metrics_json[experiment_name] = experiment_json

    new_combined_json_path = 'metrics/all_experiments_{}_{}_metrics.json'.format(experiment_key_list,
                                                                             datetime.now().strftime('%y%m%d%H%M%S'))
    with open(new_combined_json_path, 'w') as f:
        json.dump(multi_experiment_metrics_json, f, indent=4)

    return multi_experiment_metrics_json, new_combined_json_path

def generate_plots(metrics_csv):
    metrics_logs = pd.read_csv(train_csv)

    # Average train score
    plt.plot(train_logs['episode'], train_logs['smoothed_score'])
    plt.xlabel('Episode')
    plt.ylabel('Average Train Score')
    plt.show()

    # Average eval score
    plt.plot(eval_logs['episode'], eval_logs['scores_mean'])
    plt.xlabel('Episode')
    plt.ylabel('Average Eval Score')
    plt.show()

    # Average train completion rate
    plt.plot(train_logs['episode'], train_logs['smoothed_completion'])
    plt.xlabel('Episode')
    plt.ylabel('Average Train Completion Rate')
    plt.show()

    # Average eval completion rate
    plt.plot(eval_logs['episode'], eval_logs['completions_mean'])
    plt.xlabel('Episode')
    plt.ylabel('Average Eval Completion Rate')
    plt.show()


def plot_experiment_results(experiments, metrics, title, ylabel, filename, plot_label, exp_filter):
    plt.figure(figsize=(12, 7.5))

    # --- different experiment has different colours and train and val metrics have different line style for each experiment------

    metric_linestyle = {'training': '-', 'evaluation': '--'}
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']
    experiment_colour = {}
    i = 0
    x_values = np.arange(100, 4001, 100)  # Create x-axis values
    y_min = float('inf')
    y_max = float('-inf')

    for exp_name in experiments.keys():
        if exp_name in exp_filter:
            exp_key = exp_name.split("_")[1]
            experiment_colour[exp_key] = colors[i]
            i += 1

            if isinstance(metrics, list):
                for metric_name in metrics:
                    metric = metric_name.split("-")[1]
                    if "training" in metric_name:
                        train_metric_min = np.min(experiments[exp_name]["training"][metric])
                        train_metric_max = np.max(experiments[exp_name]["training"][metric])
                        plt.plot(x_values, experiments[exp_name]["training"][metric], color=experiment_colour[exp_key],
                                 linestyle=metric_linestyle["training"],
                                 label="{}-{}".format(plot_label[i - 1], "train"))
                    else:
                        eval_metric_min = np.min(experiments[exp_name]["evaluation"][metric])
                        eval_metric_max = np.max(experiments[exp_name]["evaluation"][metric])
                        plt.plot(x_values, experiments[exp_name]["evaluation"][metric],
                                 color=experiment_colour[exp_key],
                                 linestyle=metric_linestyle["evaluation"],
                                 label="{}-{}".format(plot_label[i - 1], "eval"))
            else:
                metric = metrics.split("-")[1]

                if "training" in metrics:
                    train_metric_min = np.min(experiments[exp_name]["training"][metric])
                    train_metric_max = np.max(experiments[exp_name]["training"][metric])
                    plt.plot(x_values, experiments[exp_name]["training"][metric], color=experiment_colour[exp_key],
                             linestyle=metric_linestyle["training"], label="{}-{}".format(plot_label, "train"))
                else:
                    eval_metric_min = np.min(experiments[exp_name]["evaluation"][metric])
                    eval_metric_max = np.max(experiments[exp_name]["evaluation"][metric])
                    plt.plot(x_values, experiments[exp_name]["evaluation"][metric], color=experiment_colour[exp_key],
                             linestyle=metric_linestyle["evaluation"], label="{}-{}".format(plot_label, "eval"))

    num_intervals = 10
    y_min = min(y_min, train_metric_min, eval_metric_min)
    y_max = max(y_max, train_metric_max, eval_metric_max)
    y_min = round(y_min, 2)
    y_max = round(y_max, 2)
    interval = round((y_max - y_min) / 10, 2)
    y_min = np.round(y_min - interval * (num_intervals/2-1))
    y_max = np.round(y_max + interval * (num_intervals/2-1))
    # print(y_min, y_max, np.linspace(y_min, y_max, num=10))
    plt.title(title, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xlabel('episode', fontsize=11)
    plt.legend(loc='best', ncols=3, fontsize=11)
    plt.xticks(np.arange(0, 4100, 200))
    plt.yticks(np.linspace(y_min, y_max, num=num_intervals))  # (np.arange(-1, 0, 0.05))
    plt.grid(color='grey', linestyle=':', linewidth=0.5)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, format='png', transparent=False, orientation='landscape')
    plt.show()
    plt.clf()
    plt.close()
