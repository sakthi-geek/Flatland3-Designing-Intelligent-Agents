# from stable_baselines3 import DQN, PPO, A2C
# from pettingzoo.utils.conversions import parallel_wrapper_fn
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from ray.tune.registry import register_env
# import supersuit as ss
# from stable_baselines3.common.vec_env import VecTransposeImage
from flatland.utils.rendertools import RenderTool
import matplotlib.pyplot as plt
# from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
# from flatland_3_pettingzoo_env import Flatland3PettingZoo
from flatland.envs.observations import (GlobalObsForRailEnv,
                                        LocalObsForRailEnv, TreeObsForRailEnv)
from custom_observations import CombinedLocalGlobalObs

from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator


from flatland.envs.rail_env import RailEnv
# from custom_rail_env import RailEnv
# from gym.core import Wrapper

# class FlatlandRailEnvWrapper(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#     @property
#     def unwrapped(self):
#         return self.env


def create_flatland3_env(env_config, obs_params):

    _seed = env_config["seed"]
    n_agents = env_config["n_agents"]
    width = env_config["width"]
    height = env_config["height"]
    max_num_cities = env_config["max_num_cities"]
    grid_mode = env_config["grid_mode"]
    max_rails_between_cities = env_config["max_rails_between_cities"]
    max_rail_pairs_in_city = env_config["max_rail_pairs_in_city"]
    agents = [f'train_{i}' for i in range(env_config["n_agents"])]
    obs_type = env_config["obs_type"]
    tree_max_depth = obs_params["max_tree_depth"]
    # malfunction_params = env_config["malfunction_params"]
    # malfunction_rate = malfunction_params["malfunction_rate"]
    # min_duration = malfunction_params["min_duration"]
    # max_duration = malfunction_params["max_duration"]

    ## Initialize Flatland environment
    rail_generator = sparse_rail_generator(
        max_num_cities=4,
        grid_mode=True,
        max_rails_between_cities=max_rails_between_cities,
        max_rail_pairs_in_city=max_rail_pairs_in_city,
    )
    line_generator = sparse_line_generator()
    if obs_type == 'tree':
        obs_builder = TreeObsForRailEnv(max_depth=tree_max_depth)
    elif obs_type == 'local':
        obs_builder = LocalObsForRailEnv()
    elif obs_type == 'global':
        obs_builder = GlobalObsForRailEnv()
    elif obs_type == "combined":
        obs_builder = CombinedLocalGlobalObs(tree_depth=tree_max_depth)
    else:
        raise ValueError(f"Invalid observation type: {obs_type}")
    # Malfunction parameters
    # stochastic_data = MalfunctionParameters(
    #     malfunction_rate=malfunction_rate,  #1 / 10000,  # Rate of malfunction occurence
    #     min_duration=min_duration,          #15,  # Minimal duration of malfunction
    #     max_duration=max_duration           #50  # Max duration of malfunction
    # )
    # malfunction_generator = ParamMalfunctionGen(stochastic_data)
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
    fl3_env.reset()
    fl3_env.possible_agents = [i for i in range(fl3_env.number_of_agents)]

    return fl3_env

def create_flatland3_pettingzoo_env(env_config):
    return Flatland3PettingZoo(env_config)

def render_env(env):
        # Initialize the renderer
        env_renderer = RenderTool(env)
        # Reset the environment
        env_observe = env.reset()
        # Render the environment
        env_image = env_renderer.render_env(show=False, frames=False, show_observations=False, return_image=True)

        plt.imshow(env_image)
        # ---save the environment image
        plt.savefig('flatland3_env_image_1.png')
        # ---show the environment imag
        plt.show()
        print("env_image saved")


def render_parallel_envs(envs, save_dir, env_index=-1, episode=0):
    if env_index == -1:
        env_images = envs.get_images()
        for i, env_image in enumerate(env_images):
            render_parallel_envs(envs, save_dir, env_index=i, episode=episode)
    else:
        env_images = envs.get_images()
        image_array = np.asarray(env_images[env_index])
        img = Image.fromarray(image_array)
        image_filename = f'env_{env_index}_image_episode_{episode}.png'
        image_path = os.path.join(save_dir, image_filename)
        img.save(image_path)
        print(f"env_image saved in {image_path}")


def get_flatland_3_environment(env_config, num_envs, env_setting="parallel"):
    if env_setting == "parallel":
        fl3_parallel_envs = SubprocVecEnv([Flatland3PettingZoo(env_config) for _ in range(num_envs)])
        fl3_env = fl3_parallel_envs

        # Use VecTransposeImage to transpose the image data for correct visualization
        fl3_parallel_env_images = VecTransposeImage(fl3_parallel_envs)

        # Render images for each environment
        env_images = fl3_parallel_env_images.env_method("render", mode="rgb_array")

        for idx, env_image in enumerate(env_images):
            plt.imshow(env_image)
            plt.savefig(f"env_image_{idx}.png")
            plt.show()

    else:
        fl3_env = Flatland3PettingZoo(env_config)

        # Initialize the renderer
        env_renderer = RenderTool(fl3_env)
        # Reset the environment
        env_observe = fl3_env.reset()
        # Render the environment
        env_image = env_renderer.render_env(show=False, frames=False, show_observations=False, return_image=True)

        plt.imshow(env_image)
        # ---save the environment image
        plt.savefig('env_image_1.png')
        # ---show the environment imag
        plt.show()

    return fl3_env

def get_model_class(agent_name):
    if agent_name == 'DQN':
        return DQN
    elif agent_name == 'PPO':
        return PPO
    elif agent_name == 'A2C':
        return A2C
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")


# Create the parallel PettingZoo environment
def create_parallel_pettingzoo_env(env):
    env = parallel_wrapper_fn(env)
    return env

def prep_parallel_envs_for_stable_baselines3(env, num_envs=1, num_cpus=1, base_class='stable_baselines3'):
    # create metadata dictionary
    metadata = {"is_parallelizable": True}
    env.metadata = metadata
    env = aec_to_parallel(env)
    # env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=num_cpus, base_class=base_class)
    return env

#----------------------------------------------------------------------------------------------------
def env_creator(env_config):  ##--- still in progress ------
    env = Flatland3PettingZoo(env_config)
    env = create_parallel_pettingzoo_env(env)
    return env

def prep_env_for_rllib(env_name, env_config):   ##--- still in progress ------

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(env_config)))
    env = ParallelPettingZooEnv(env_creator({}))
    return env


def calculate_completion_rate(env):
    num_agents = len(env.agents)
    num_agents_done = sum([1 for agent in env.agents if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE])
    completion_rate = (num_agents_done / num_agents) * 100
    return completion_rate


def calculate_collision_rate(env):
    num_agents = len(env.agents)
    num_agents_collided = sum([1 for agent in env.agents if agent.status == env.TrainState.DEADLOCK])
    collision_rate = (num_agents_collided / num_agents) * 100
    return collision_rate

def calculate_average_travel_time(env, max_time_steps):
    travel_times = [
        max_time_steps - agent.handle_time for agent in env.agents if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE
    ]
    if len(travel_times) > 0:
        avg_travel_time = sum(travel_times) / len(travel_times)
    else:
        avg_travel_time = 0
    return avg_travel_time

def calculate_average_delay(env, shortest_travel_times):  # not completed  - inprogress
    shortest_travel_times = []
    rail_map = sparse_to_dense(env.rail)
    for agent in env.agents:
        shortest_path_length = env.shortest_path(agent.handle)[1]
        if shortest_path_length is not None:
            shortest_travel_times.append(shortest_path_length)
        else:
            shortest_travel_times.append(float("inf"))

    delays = [
        (max_time_steps - agent.handle_time) - shortest_travel_times[i] for i, agent in enumerate(env.agents)
        if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE
    ]
    if len(delays) > 0:
        avg_delay = sum(delays) / len(delays)
    else:
        avg_delay = 0
    return avg_delay

def calculate_malfunction_rate(env):  # - in progress
    num_agents = len(env.agents)
    num_malfunctions = sum([agent.total_malfunction for agent in env.agents])
    malfunction_rate = (num_malfunctions / num_agents) * 100
    return malfunction_rate

def plot_results(metrics_callback, exp_params):
    # Plot reward, episode length, and training time
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(metrics_callback.rewards)
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(metrics_callback.episode_lengths)
    axs[1].set_title("Episode Lengths")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Length")

    axs[2].plot(metrics_callback.training_times)
    axs[2].set_title("Training Times")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(exp_params["eval_save_path"], "training_metrics.png"))
    plt.show()

    # Plot completion rate and collision rate
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(metrics_callback.completion_rates)
    axs[0].set_title("Completion Rates")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Completion Rate")

    axs[1].plot(metrics_callback.collision_rates)
    axs[1].set_title("Collision Rates")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Collision Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(exp_params["eval_save_path"], "performance_metrics.png"))
    plt.show()



