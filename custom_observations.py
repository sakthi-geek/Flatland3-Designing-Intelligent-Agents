import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

class CustomObservationBuilder(ObservationBuilder):
    """
    Custom Observation Builder class with three custom observations:
    1. Agent-specific distance map.
    2. Number of transitions at the current cell.
    3. Overlapping predictions with other agents.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def reset(self):
        self.tree_obs = TreeObsForRailEnv(max_depth=0)
        self.tree_obs.set_env(self.env)
        self.tree_obs.reset()

    def get(self, handle):
        agent = self.env.agents[handle]

        # 1. Agent-specific distance map
        distance_map = self.tree_obs.env.distance_map.get()[handle]

        # 2. Number of transitions at the current cell
        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # 3. Overlapping predictions with other agents
        self.predictions = self.predictor.get()
        overlaps = self.get_overlapping_predictions(handle)

        return np.array([distance_map, num_transitions, overlaps])

    def get_overlapping_predictions(self, handle):
        predicted_pos = {}
        for t in range(len(self.predictions[0])):
            pos_list = []
            for a in range(self.env.get_num_agents()):
                pos_list.append(self.predictions[a][t][1:3])
            predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})

        overlaps = 0
        for _idx in range(10):
            if predicted_pos[_idx][handle] in np.delete(predicted_pos[_idx], handle, 0):
                overlaps += 1

        return overlaps

def create_flatland_environment(env_params):
    # Create the Predictor
    CustomPredictor = ShortestPathPredictorForRailEnv(10)

    # Pass the Predictor to the observation builder
    CustomObsBuilder = CustomObservationBuilder(CustomPredictor)

    env = RailEnv(
        width=env_params['width'],
        height=env_params['height'],
        rail_generator=env_params['rail_generator'],
        line_generator=env_params['line_generator'],
        number_of_agents=env_params['number_of_agents'],
        obs_builder_object=CustomObsBuilder,
    )
    return env

# Define the environment parameters
env_params = {
    'width': 30,
    'height': 30,
    'number_of_agents': 3,
    'rail_generator': sparse_rail_generator(),
    'line_generator': sparse_line_generator(),
}

# Create the environment
env = create_flatland_environment(env_params)