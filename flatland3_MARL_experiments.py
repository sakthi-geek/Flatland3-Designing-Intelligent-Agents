import json
import os
import glob
import helper
from pprint import pprint


def run_experiment(exp_name, exp_params, train_env_config, eval_env_config):
    print("Running experiment: ", exp_name, " with below params: ")
    pprint(exp_params, indent=4)

    helper.initial_prep_for_experiment(exp_name, exp_params)

    #--- env params ------
    env_config = exp_params["env_config"]
    obs_params = exp_params["obs_params"]
    seed = env_config["seed"]

    # Create environment
    train_fl3_env = helper.create_flatland3_env(train_env_config, obs_params)
    eval_fl3_env = helper.create_flatland3_env(eval_env_config, obs_params)
    print("Created environment")

    # observation parameters
    obs_params = exp_params["obs_params"]

    # Train the agent
    os.environ["OMP_NUM_THREADS"] = str(exp_params.get("num_threads"))
    helper.train_agent(exp_name, exp_params, train_fl3_env, eval_fl3_env, obs_params)

    ###--- convert csv to json ------
    # csv_file_path = 'metrics/experiment_1_230508021106_metrics.csv'
    # json_file_path = 'metrics/experiment_1_230508021106_metrics.json'
    # helper.convert_csv_to_json(csv_file_path, json_file_path, type='custom')
    # sdsd

    # Combine all experiments csv files into a single dataframe
    csv_files = glob.glob('metrics/experiment_*_*_metrics.csv')
    combined_metrics_df = helper.combine_multiple_experiments_csv(csv_files)
    print("Saved all experiments metrics into a single csv file")

    #  Combine all experiments json files into a single dataframe
    json_files = glob.glob('metrics/experiment_*_*_metrics.json')
    multi_experiments_output_json, multi_experiments_output_json_path = helper.combine_multiple_experiments_json(json_files)
    print("Saved all experiments metrics into a single json file")

    # Manual loading of multi experiments output json file
    # multi_experiments_output_json_path = "metrics/experiments_['1', '2', '3', '4', '5']_230508132715_metrics.json"
    # with open(multi_experiments_output_json_path) as f:
    #     multi_experiments_output_json = json.load(f)

    # pprint(multi_experiments_output_json)
    # Plot all experiments
    score_metrics = ["training-smoothed_score", "evaluation-eval_score"]
    completion_metrics = ["training-smoothed_completion", "evaluation-eval_completion"]
    travel_time_metrics = ["training-average_travel_time", "evaluation-eval_travel_time"]
    deadlock_metrics = ["training-normalized_deadlocks", "evaluation-eval_deadlocks"]
    history_metrics_dict = {"score": score_metrics, "completion": completion_metrics,
                            "travel_time": travel_time_metrics, "deadlocks": deadlock_metrics}

    for metric, history_metrics in history_metrics_dict.items():
        # ---------------------- metric comparision plots ----------------------------------
        # -- malfunction_experiments - 1,3,4 ------------
        experiments_filter = ["experiment_1", "experiment_3", "experiment_4"]
        experiments_key_filter = ["1", "3", "4"]
        plot_title = "30x30_grid-4_trains-malfunction_experiments-[1,3,4]-{}".format(metric)
        plot_label = ["malfunction_0", "malfunction_0.0025", "malfunction_0.005"]  # 1/400=0.0025 , 1/200=0.005
        file_path = "experiments/30x30_grid-4_trains-malfunction_experiments_{}_{}.png".format(experiments_key_filter,
                                                                                               metric)
        helper.plot_experiment_results(multi_experiments_output_json, history_metrics, plot_title, metric, file_path,
                                plot_label, experiments_filter)

        # -- observation_experiments - 1,2,9 ------------
        experiments_filter = ["experiment_1", "experiment_2", "experiment_9"]
        experiments_key_filter = ["1", "2", "9"]
        plot_title = "30x30_grid-4_trains-observation_experiments-[1,2,9]-{}".format(metric)
        plot_label = ["tree_2-10-20", "tree_3-10-20", "tree_3-20-30"]
        file_path = "experiments/30x30_grid-4_trains-observation_experiments_{}_{}.png".format(experiments_key_filter,
                                                                                               metric)
        helper.plot_experiment_results(multi_experiments_output_json, history_metrics, plot_title, metric, file_path,
                                plot_label, experiments_filter)

        # -- num_agents_experiments - 1,5,6 ------------
        experiments_filter = ["experiment_1", "experiment_5", "experiment_6"]
        experiments_key_filter = ["1", "5", "6"]
        plot_title = "30x30_grid-4_trains-num_agents_experiments-[1,5,6]-{}".format(metric)
        plot_label = ["trains_4", "trains_7", "trains_10"]
        file_path = "experiments/30x30_grid-tree_obs-num_agents_experiments_{}_{}.png".format(experiments_key_filter,
                                                                                               metric)
        helper.plot_experiment_results(multi_experiments_output_json, history_metrics, plot_title, metric, file_path,
                                plot_label, experiments_filter)

        # -- env_complexity_experiments - 6,7,8 ------------
        experiments_filter = ["experiment_6", "experiment_7", "experiment_8"]
        experiments_key_filter = ["6", "7", "8"]
        plot_title = "30x30_grid-10_trains-env_complexity_experiments-[6,7,8]-{}".format(metric)
        plot_label = ["env_30-30-2-2-2", "env_35-35-4-2-2", "env_35-35-4-4-4"]
        file_path = "experiments/tree_obs-10_trains-env_complexity_experiments_{}_{}.png".format\
                                                                    (experiments_key_filter, metric)
        helper.plot_experiment_results(multi_experiments_output_json, history_metrics, plot_title, metric, file_path,
                                plot_label, experiments_filter)










if __name__ == "__main__":
    # load experiments from json file
    with open('flatland3_MARL_experiments.json') as f:
        experiments = json.load(f)

    malfunction_experiments = ["experiment_1", "experiment_4", "experiment_3"]   # stochasticity
    observation_experiments = ["experiment_1", "experiment_2", "experiment_9"]  # observation tree depth
    num_agents_experiments = ["experiment_1", "experiment_5", "experiment_6"]   # number of agents
    env_complexity_experiments = ["experiment_6", "experiment_7", "experiment_8"] # env complexity

    #---- future experiments ----
    model_experiments = ["experiment_10", "experiment_11", "experiment_12"]
    custom_observation_experiments = ["experiment_13", "experiment_14", "experiment_15"]
    real_world_scenarios_experiments = ["experiment_16", "experiment_17", "experiment_18"]


    # experiments_to_run = ["experiment_5", "experiment_6"]
    experiments_to_run = ["experiment_8"]

    print(experiments_to_run)

    for exp_name, exp_params in experiments.items():
        # run experiment if it is in the list of experiments to run
        if exp_name in experiments_to_run:
            run_experiment(exp_name, exp_params, exp_params["env_config"], exp_params["env_config"])