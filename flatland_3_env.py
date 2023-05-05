from typing import Tuple, Callable, Optional, Any, List
from flatland.core.grid.grid4_utils import get_new_position
# from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.observations import (
    TreeObsForRailEnv,
    GlobalObsForRailEnv,
    LocalObsForRailEnv,
)
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool
import numpy as np
import matplotlib.pyplot as plt


def create_flatland_environment(
    width: int,
    height: int,
    num_agents: int,
    max_num_cities: int = 5,
    grid_mode: bool = False,
    max_rails_between_cities: int = 4,
    max_rail_pairs_in_city: int = 4,
    seed: int = 0,
    observation_type: str = "tree",
    custom_observation: Optional[Callable] = None,
):
    
    # Define the parameters for the environment
    rail_generator = sparse_rail_generator(
        max_num_cities=max_num_cities,
        grid_mode=grid_mode,
        max_rails_between_cities=max_rails_between_cities,
        max_rail_pairs_in_city=max_rail_pairs_in_city,
        seed=seed,
    )

    # Set the line generator
    line_generator = sparse_line_generator()

    if observation_type == "tree":
        observation_builder = TreeObsForRailEnv(max_depth=2)
    elif observation_type == "global":
        observation_builder = GlobalObsForRailEnv()
    elif observation_type == "local":
        observation_builder = LocalObsForRailEnv()
    elif observation_type == "custom" and custom_observation is not None:
        observation_builder = custom_observation
    else:
        raise ValueError("Invalid observation type specified.")

    rail_env = RailEnv(
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=num_agents,
        obs_builder_object=observation_builder,
    )

    # Initialize the renderer
    env_renderer = RenderTool(rail_env)

    # Reset the environment
    obs, info = rail_env.reset()

    # Render the environment
    env_image = env_renderer.render_env(show=False, frames=False, show_observations=False, return_image=True)
    
    plt.imshow(env_image)
    #---save the environment image
    plt.savefig('env_image_1.png')
    #---show the environment imag
    plt.show()
    #--------------------------------------------

    return rail_env


env = create_flatland_environment(
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

# import super_agent as ss
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import MlpPolicy


# env = flatland_env.env(environment=rail_env, use_renderer=True)
# seed = 11
# env.reset(random_seed=seed)
# step = 0
# ep_no = 0
# frame_list = []
# while ep_no < total_episodes:
#     for agent in env.agent_iter():
#         obs, reward, done, info = env.last()
#         # act = env_generators.get_shortest_path_action(env.environment, get_agent_handle(agent))
#         act = 2
#         all_actions_pettingzoo_env.append(act)
#         env.step(act)
#         frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
#         step += 1

# env = flatland_env.parallel_env(environment=rail_env, use_renderer=False)

# env_steps = 1000  # 2 * env.width * env.height  # Code uses 1.5 to calculate max_steps
# rollout_fragment_length = 50
# env = ss.pettingzoo_env_to_vec_env_v0(env)
# env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

# model = PPO(MlpPolicy, env, tensorboard_log=f"/tmp/{experiment_name}", verbose=3, gamma=0.95, 
#     n_steps=rollout_fragment_length, ent_coef=0.01, 
#     learning_rate=5e-5, vf_coef=1, max_grad_norm=0.9, gae_lambda=1.0, n_epochs=30, clip_range=0.3,
#     batch_size=150, seed=seed)
# train_timesteps = 100000
# model.learn(total_timesteps=train_timesteps)
# model.save(f"policy_flatland_{train_timesteps}")