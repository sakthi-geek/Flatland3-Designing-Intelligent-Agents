from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv

class CombinedLocalGlobalObs:
    def __init__(self, tree_depth=2, local_radius=3):
        self.local_obs_builder = TreeObsForRailEnv(max_depth=tree_depth)
        self.global_obs_builder = GlobalObsForRailEnv()
        self.local_radius = local_radius

    def set_env(self, env):
        self.local_obs_builder.set_env(env)
        self.global_obs_builder.set_env(env)

    def reset(self):
        self.local_obs_builder.reset()
        self.global_obs_builder.reset()

    def get(self, handle):
        local_obs = self.local_obs_builder.get(handle)
        global_obs = self.global_obs_builder.get(handle)

        # Crop local area from the global observation
        local_area = global_obs[
            handle,
            :,
            self.local_radius * -1 : self.local_radius + 1,
            self.local_radius * -1 : self.local_radius + 1,
        ]

        # Flatten the local and global observations and concatenate them
        combined_obs = np.concatenate((local_obs, local_area.flatten()))
        return combined_obs