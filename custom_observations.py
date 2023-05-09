from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.core.env_observation_builder import ObservationBuilder
import numpy as np

class CombinedLocalGlobalObs(ObservationBuilder):
    def __init__(self, tree_depth=2, predictor=ShortestPathPredictorForRailEnv(), local_radius=3):
        super().__init__()
        self.tree_obs_builder = TreeObsForRailEnv(max_depth=tree_depth, predictor=predictor)
        self.global_obs_builder = GlobalObsForRailEnv()
        self.local_radius = local_radius

        self.observation_dim = self.tree_obs_builder.observation_dim
        self.max_depth = self.tree_obs_builder.max_depth
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = self.tree_obs_builder.predictor
        self.location_has_target = None

    def set_env(self, env):
        self.tree_obs_builder.set_env(env)
        self.global_obs_builder.set_env(env)

    def reset(self):
        self.tree_obs_builder.reset()
        self.global_obs_builder.reset()

    def get(self, handle):
        tree_obs = self.tree_obs_builder.get(handle)
        global_obs = self.global_obs_builder.get(handle)

        # Crop local area from the global observation
        local_area = global_obs[0][
             self.local_radius * -1: self.local_radius + 1,
             self.local_radius * -1: self.local_radius + 1,
             :,
        ]

        # Flatten the local and global observations and concatenate them
        combined_obs = np.concatenate((tree_obs, local_area.flatten()))
        return combined_obs

    def get_many(self, handles):
        all_tree_obs = self.tree_obs_builder.get_many(handles)
        all_global_obs = self.global_obs_builder.get_many(handles)
        # print(all_global_obs)

        # Crop local area from the global observation dict -{handle: global_obs for handle, global_obs in all_global_obs.items()}
        all_local_areas = {}
        for handle, global_obs in all_global_obs.items():
            print(type(global_obs), len(global_obs))
            print(global_obs[0].shape)
            all_local_areas[handle] = global_obs[0][
                self.local_radius * -1 : self.local_radius + 1,
                self.local_radius * -1 : self.local_radius + 1,
                :,
            ]

        # combine both the dicts - all_tree_obs and all_local_areas
        all_combined_obs = {}
        for handle, tree_obs in all_tree_obs.items():
            all_combined_obs[handle] = np.concatenate((tree_obs, all_local_areas[handle].flatten()))

        return all_combined_obs



