import logging
import yaml

from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformIntegerParameterRange, GridSearch)

from clearml.automation.optuna import OptimizerOptuna  # noqa
aSearchStrategy = OptimizerOptuna

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--job_id", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="config_sweep_clearML.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as yml:
        sweep_config = yaml.load(yml, Loader=yaml.SafeLoader)

    base_task_id = args.job_id
    hyper_parameters=[DiscreteParameterRange(p, values=sweep_config["parameters"][p]["values"]) for p in sweep_config["parameters"]]
    objective_metric_title = sweep_config["metric"]["name"]
    objective_metric_series = sweep_config["metric"]["name"]
    objective_metric_sign=sweep_config["metric"]["goal"]

    # execution_queue = '1xGPU'

    # Example use case:
    optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=base_task_id,
        # here we define the hyper-parameters to optimize
        # Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
        # For Example, here we see in the base experiment a section Named: "General"
        # under it a parameter named "batch_size", this becomes "General/batch_size"
        # If you have `argparse` for example, then arguments will appear under the "Args" section,
        # and you should instead pass "Args/batch_size"
        hyper_parameters=hyper_parameters,
        # this is the objective metric we want to maximize/minimize
        objective_metric_title=objective_metric_title,
        objective_metric_series=objective_metric_series,
        # now we decide if we want to maximize it or minimize it (accuracy we maximize)
        objective_metric_sign=objective_metric_sign,
        # let us limit the number of concurrent experiments,
        # this in turn will make sure we do dont bombard the scheduler with experiments.
        # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
        max_number_of_concurrent_tasks=1,
        # this is the optimizer class (actually doing the optimization)
        # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
        # more are coming soon...
        optimizer_class=GridSearch,
        # # Select an execution queue to schedule the experiments for execution
        # execution_queue=execution_queue,
        # If specified all Tasks created by the HPO process will be created under the `spawned_project` project
        spawn_project=None,  # 'HPO spawn project',
        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=None,  # 5,
        # # Optional: Limit the execution time of a single experiment, in minutes.
        # # (this is optional, and if using  OptimizerBOHB, it is ignored)
        # time_limit_per_job=10.,
        # Check the experiments every 12 seconds is way too often, we should probably set it to 5 min,
        # assuming a single experiment is usually hours...
        pool_period_min=0.2,
        # # set the maximum number of jobs to launch for the optimization, default (None) unlimited
        # # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
        # # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
        # total_max_jobs=10,
        # # set the minimum number of iterations for an experiment, before early stopping.
        # # Does not apply for simple strategies such as RandomSearch or GridSearch
        # min_iteration_per_job=10,
        # # Set the maximum number of iterations for an experiment to execute
        # # (This is optional, unless using OptimizerBOHB where this is a must)
        # max_iteration_per_job=30,
    )

    # report every 12 seconds, this is way too often, but we are testing here 
    optimizer.set_report_period(0.2)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    optimizer.start_locally()
    # You can also use the line below instead to run all the optimizer tasks locally, without using queues or agent
    # an_optimizer.start_locally(job_complete_callback=job_complete_callback)
    # set the time limit for the optimization process (2 hours)
    optimizer.set_time_limit(in_minutes=120.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()

    print('We are done, good bye')


if __name__ == "__main__":
    main()