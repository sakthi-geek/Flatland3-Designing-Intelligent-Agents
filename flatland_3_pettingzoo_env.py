import numpy as np
from flatland.core.env_observation_builder import RailAgentStatus
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import (GlobalObsForRailEnv,
                                        LocalObsForRailEnv, TreeObsForRailEnv)
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from matplotlib import pyplot as plt
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from custom_observations import CombinedLocalGlobalObs


class Flatland3PettingZoo(AECEnv):

    def __init__(self, env_config):
        # Set environment properties
        self._seed = env_config["seed"]
        self.n_agents = env_config["n_agents"]
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.max_num_cities = env_config["max_num_cities"]
        self.grid_mode = env_config["grid_mode"]
        self.max_rails_between_cities = env_config["max_rails_between_cities"]
        self.max_rail_pairs_in_city = env_config["max_rail_pairs_in_city"]
        self.agents = [f'agent_{i}' for i in range(env_config["n_agents"])]
        self.obs_type = env_config["obs_type"]
        self.tree_max_depth = env_config["tree_max_depth"]

        # Initialize agent selector
        self._agent_selector = agent_selector(self.agents)
        self._agent_order = list(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Initialize Flatland environment
        self._init_flatland_env()

        # Set action and observation spaces
        self.action_spaces, self.observation_spaces = self._get_spaces()

    def _init_flatland_env(self):
        rail_generator = sparse_rail_generator(
            max_num_cities=self.max_num_cities,
            grid_mode=self.grid_mode,
            max_rails_between_cities=self.max_rails_between_cities,
            max_rails_in_city=self.max_rail_pairs_in_city,
            seed=self._seed
        )

        line_generator = sparse_line_generator()

        if self.obs_type == 'tree':
            obs_builder = TreeObsForRailEnv(max_depth=self.tree_max_depth)
        elif self.obs_type == 'local':
            obs_builder = LocalObsForRailEnv()
        elif self.obs_type == 'global':
            obs_builder = GlobalObsForRailEnv()
        elif self.obs_type == "combined":
            obs_builder = CombinedLocalGlobalObs(tree_depth=self.tree_max_depth)
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

        self.flatland_env = RailEnv(
            width=self.width,
            height=self.height,
            rail_generator=rail_generator,
            line_generator=line_generator,
            number_of_agents=self.n_agents,
            obs_builder_object=obs_builder,
            random_seed=self._seed
        )

    def _get_spaces(self):
        action_spaces = {
            agent: self.flatland_env.action_space for agent in self.agents
        }
        observation_spaces = {
            agent: self.flatland_env.observation_space for agent in self.agents
        }
        return action_spaces, observation_spaces

    def reset(self):
        self._agent_selector.reinit(self.agents)
        self._agent_order = list(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.flatland_env.reset()
        return self._observe()

    def _observe(self):
        observations = self.flatland_env.get_many_obs()
        return {
            agent: observations[idx] for idx, agent in enumerate(self.agents)
        }

    def step(self, action):
        _, _, done, _ = self.flatland_env.step({self.agent_selection: action})
        self._was_done_step = done
        next_agent = self._agent_selector.next()
        self.agent_selection = next_agent
        return self._observe()

    def render(self, mode='human'):
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
            if agent.status == RailAgentStatus.DONE_REMOVED:
                completed_trains += 1
        completion_rate = completed_trains / len(self.flatland_env.agents)
        return completion_rate

    def get_collision_rate(self):
        collided_trains = 0
        for agent in self.flatland_env.agents:
            if agent.status == RailAgentStatus.CRASHED:
                collided_trains += 1
        collision_rate = collided_trains / len(self.flatland_env.agents)
        return collision_rate