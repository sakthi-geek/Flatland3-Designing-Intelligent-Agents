import numpy as np
from flatland.envs.step_utils.states import TrainState
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import (GlobalObsForRailEnv,
                                        LocalObsForRailEnv, TreeObsForRailEnv)
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from matplotlib import pyplot as plt
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector
from typing import Dict, Any, List
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import functools
from gymnasium import wrappers

from custom_observations import CombinedLocalGlobalObs


class Flatland3PettingZoo(gym.Env):

    def __init__(self, env_config):
        super().__init__()
        # Set environment properties
        self._seed = env_config["seed"]
        self.n_agents = env_config["num_agents"]
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.max_num_cities = env_config["max_num_cities"]
        self.grid_mode = env_config["grid_mode"]
        self.max_rails_between_cities = env_config["max_rails_between_cities"]
        self.max_rail_pairs_in_city = env_config["max_rail_pairs_in_city"]
        self.agents = [f'train_{i}' for i in range(env_config["num_agents"])]
        self.obs_type = env_config["obs_type"]
        self.tree_max_depth = env_config["tree_max_depth"]

        ## Initialize agent selector
        self._agent_selector = agent_selector(self.agents)
        self._agent_order = list(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.possible_agents = [f'train_{i}' for i in range(env_config["num_agents"])]

        ## Initialize Flatland environment
        self._init_flatland_env()

        # self.observation_space = self.flatland_env.obs_dict[0]
        # self.action_space = self.flatland_env.action_space[0]

        # print(self.flatland_env.obs_dict)
        # self.flatland_env.possible_agents = [f'train_{i}' for i in range(self.n_agents)]
        # self.flatland_env.observation_space = self.flatland_env.obs_dict
        # print(self.flatland_env.observation_space)

        ## Set action and observation spaces
        self.action_spaces, self.observation_spaces = self._get_spaces()

        # self.action_space = self.action_spaces(self._agent_selector)
        # self.observation_space = self.observation_spaces(self._agent_selector)


    def _init_flatland_env(self):
        rail_generator = sparse_rail_generator(
            max_num_cities=self.max_num_cities,
            grid_mode=self.grid_mode,
            max_rails_between_cities=self.max_rails_between_cities,
            max_rail_pairs_in_city=self.max_rail_pairs_in_city,
        )

        line_generator = sparse_line_generator()

        if self.obs_type == 'tree':
            obs_builder = TreeObsForRailEnv(max_depth=self.tree_max_depth)
        elif self.obs_type == 'local':
            obs_builder = LocalObsForRailEnv(view_width=self.width//3, view_height=self.height//3, center=0)
        elif self.obs_type == 'global':
            obs_builder = GlobalObsForRailEnv()
        elif self.obs_type == "combined":
            obs_builder = CombinedLocalGlobalObs(tree_depth=self.tree_max_depth)
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

        # Malfunction parameters
        # stochastic_data = MalfunctionParameters(
        #     malfunction_rate=1 / 10000,  # Rate of malfunction occurence
        #     min_duration=15,  # Minimal duration of malfunction
        #     max_duration=50  # Max duration of malfunction
        # )
        # malfunction_generator = ParamMalfunctionGen(stochastic_data)

        malfunction_generator = None

        self.flatland_env = RailEnv(
            width=self.width,
            height=self.height,
            rail_generator=rail_generator,
            line_generator=line_generator,
            malfunction_generator=malfunction_generator,
            number_of_agents=self.n_agents,
            obs_builder_object=obs_builder,
            random_seed=self._seed
        )



    def _get_spaces(self):

        action_spaces = {agent: gym.spaces.Discrete(self.flatland_env.action_space[0]) for agent in self.agents}

        if self.obs_type == 'tree':
            n_features_per_node = self.flatland_env.obs_builder.observation_dim
            print(n_features_per_node)
            n_nodes = sum([np.power(4, i) for i in range(self.tree_max_depth + 1)])
            print(n_nodes)
            state_size = n_features_per_node * n_nodes
            print(state_size)
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,))
            observation_spaces = {agent: obs_space for agent in self.agents}
        elif self.obs_type == 'local':
            view_width = self.width // 3
            view_height = self.height // 3
            state_size = view_height * (2 * view_width + 1) * (16 + 2 + 2 + 4)
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,))
            observation_spaces = {agent: obs_space for agent in self.agents}
        elif self.obs_type == 'global':
            state_size = self.height * self.width * (16 + 5 + 2)
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,))
            observation_spaces = {agent: obs_space for agent in self.agents}
        elif self.obs_type == "combined":
            n_features_per_node = self.flatland_env.obs_builder.observation_dim
            n_nodes = sum([np.power(4, i) for i in range(self.tree_max_depth + 1)])
            tree_state_size = n_features_per_node * n_nodes
            local_area_size = (2 * local_radius + 1) * (2 * local_radius + 1) * (16 + 5 + 2)
            combined_state_size = tree_state_size + local_area_size
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(combined_state_size,))
            observation_spaces = {agent: obs_space for agent in self.agents}

        return action_spaces, observation_spaces

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        print(self.observation_spaces[agent])
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        print(self.action_spaces[agent])
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        # self.agents = copy(self.possible_agents)
        self._agent_selector.reinit(self.agents)
        self._agent_order = list(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.flatland_env.reset()
        return self._observe()

    def _observe(self):
        observations = self.flatland_env._get_observations()
        return {
            agent: observations[idx] for idx, agent in enumerate(self.agents)
        }

    def step(self, action_dict):
        observations, rewards, done, info = self.flatland_env.step(action_dict)  # {self.agent_selection: action}
        self._was_done_step = done
        next_agent = self._agent_selector.next()
        self.agent_selection = next_agent
        return self._observe()

    def render(self, mode='rgb_array'):
        return self.flatland_env.render(mode)

    def close(self):
        self.flatland_env.close()

    def seed(self, seed=None):
        self._seed = seed
        self._init_flatland_env()

    def last(self):
        return self.agent_selection

    def state(self):
        return self.flatland_env.get_state()

    @property
    def dones(self):
        dones = self.flatland_env.get_done_agents()
        return {
            agent: dones[idx] for idx, agent in enumerate(self.agents)
        }

    @property
    def rewards(self):
        rewards = self.flatland_env.get_agent_rewards()
        return {
            agent: rewards[idx] for idx, agent in enumerate(self.agents)
        }

    @property
    def infos(self):
        infos = self.flatland_env.get_agent_infos()
        return {
            agent: infos[idx] for idx, agent in enumerate(self.agents)
        }

    def all_done(self):
        return np.all(list(self.dones.values()))
    
    def get_completion_rate(self):
        completed_trains = 0
        for agent in self.flatland_env.agents:
            if agent.status == TrainState.DONE_REMOVED:
                completed_trains += 1
        completion_rate = completed_trains / len(self.flatland_env.agents)
        return completion_rate

    def get_collision_rate(self):
        collided_trains = 0
        for agent in self.flatland_env.agents:
            if agent.status == TrainState.CRASHED:
                collided_trains += 1
        collision_rate = collided_trains / len(self.flatland_env.agents)
        return collision_rate