""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 

import math
import sys
import time

import numpy as np
from matplotlib import pyplot as plt

from BagLearner import BagLearner
from DTLearner import DTLearner
from RTLearner import RTLearner


def get_RMSE(pred_y, test_y):
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    return rmse

def get_RSquared(pred_y, test_y):
    correlation = np.corrcoef(pred_y, test_y)[0, 1]
    return correlation**2

def plot_data(data, legend, xlabel, ylabel, title, filename, xticks=None):
    plt.clf()
    plt.plot(data.T)
    plt.legend(legend)
    if xticks:
        plt.xticks([i for i in range(len(xticks))], labels=xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(visible=True)
    plt.savefig(filename)
    pass

def experiment_1(train_x, train_y, test_x, test_y):
    leaf_sizes = [i for i in range(1, 51)]
    train_rmses = []
    test_rmses = []

    for leaf_size in leaf_sizes:
        learner = DTLearner(leaf_size)
        learner.add_evidence(train_x, train_y)

        # train_rmses
        pred_train_y = learner.query(train_x)
        train_rmse = get_RMSE(pred_train_y, train_y)
        train_rmses.append(train_rmse)

        # test rmses
        pred_test_y = learner.query(test_x)
        test_rmse = get_RMSE(pred_test_y, test_y)
        test_rmses.append(test_rmse)

    train_rmses = np.array(train_rmses)
    test_rmses = np.array(test_rmses)
    rmses = np.array([train_rmses, test_rmses])

    legend = ["in sample", "out of sample"]
    xlabel = "leaf size"
    ylabel = "RMSE"
    title = "Experiment 1: Overfitting wrt leaf size in DTLearner"
    save_path = "images/experiment_1.png"
    plot_data(rmses, legend, xlabel, ylabel, title, save_path)


def experiment_2(train_x, train_y, test_x, test_y):
    leaf_sizes = [i for i in range(1, 51)]
    num_bags = 20
    train_rmses = []
    test_rmses = []

    for leaf_size in leaf_sizes:
        learner = BagLearner(
            learner=DTLearner, kwargs={"leaf_size": leaf_size}, bags=num_bags
        )
        learner.add_evidence(train_x, train_y)

        # train_rmses
        pred_train_y = learner.query(train_x)
        train_rmse = get_RMSE(pred_train_y, train_y)
        train_rmses.append(train_rmse)

        # test rmses
        pred_test_y = learner.query(test_x)
        test_rmse = get_RMSE(pred_test_y, test_y)
        test_rmses.append(test_rmse)

    train_rmses = np.array(train_rmses)
    test_rmses = np.array(test_rmses)
    rmses = np.array([train_rmses, test_rmses])
    legend = ["in sample", "out of sample"]
    xlabel = "leaf size"
    ylabel = "RMSE"
    title = f"Experiment 2: \nOverfitting wrt leaf size in BagLearner (#bags = {num_bags})"
    save_path = "images/experiment_2.png"
    plot_data(rmses, legend, xlabel, ylabel, title, save_path)


def experiment_3_metric_1_RSquared(train_x, train_y, test_x, test_y):
    leaf_sizes = [i for i in range(1, 51)]
    rmses = {"DT_train_r2": [], "DT_test_r2": [], "RT_train_r2": [], "RT_test_r2": []}

    for leaf_size in leaf_sizes:
        # get DTLearner metrics
        dt_learner = DTLearner(leaf_size)
        dt_learner.add_evidence(train_x, train_y)

        # train_r2s
        pred_train_y = dt_learner.query(train_x)
        train_r2 = get_RSquared(pred_train_y, train_y)
        rmses["DT_train_r2"].append(train_r2)

        # test r2s
        pred_test_y = dt_learner.query(test_x)
        test_r2 = get_RSquared(pred_test_y, test_y)
        rmses["DT_test_r2"].append(test_r2)

        # get RTLearner metrics
        rt_learner = RTLearner(leaf_size)
        rt_learner.add_evidence(train_x, train_y)

        # train_r2s
        pred_train_y = rt_learner.query(train_x)
        train_r2 = get_RSquared(pred_train_y, train_y)
        rmses["RT_train_r2"].append(train_r2)

        # test r2s
        pred_test_y = rt_learner.query(test_x)
        test_r2 = get_RSquared(pred_test_y, test_y)
        rmses["RT_test_r2"].append(test_r2)

    in_sample_data = np.array([rmses["DT_train_r2"], rmses["RT_train_r2"]])
    legend = ["DTLearner", "RTLearner"]
    out_sample_data = np.array([rmses["DT_test_r2"], rmses["RT_test_r2"]])
    xlabel = "leaf size"
    ylabel = "R^2"

    # plot in-sample results
    title = f"Experiment 3, Metric 1 \nDTLearner vs RTLearner using R-Squared (in-sample)"
    save_path = "images/experiment_3_metric_1_insample.png"
    plot_data(in_sample_data, legend, xlabel, ylabel, title, save_path)

    # plot out-of-sample results
    title = f"Experiment 3, Metric 1 \nDTLearner vs RTLearner using R-Squared (out-of-sample)"
    save_path = "images/experiment_3_metric_1_outsample.png"
    plot_data(out_sample_data, legend, xlabel, ylabel, title, save_path)


def experiment_3_metric_2_traintime(train_x, train_y, test_x, test_y):
    dataset_sizes = [i for i in range(10, len(train_x), 25)]
    leaf_size = 1
    dt_learner_train_times = []
    rt_learner_train_times = []
    print(len(train_x))

    for dataset_size in dataset_sizes:
        # get DTLearner metrics
        dt_learner = DTLearner(leaf_size=leaf_size)
        train_x_subset = train_x[:dataset_size]
        train_y_subset = train_y[:dataset_size]

        start = time.time()
        dt_learner.add_evidence(train_x_subset, train_y_subset)
        train_time = time.time() - start
        dt_learner_train_times.append(train_time)

        # get RTLearner metrics
        rt_learner = RTLearner(leaf_size=leaf_size)
        start = time.time()
        rt_learner.add_evidence(train_x_subset, train_y_subset)
        train_time = time.time() - start
        rt_learner_train_times.append(train_time)

    # plot runtime
    runtimes = np.array([dt_learner_train_times, rt_learner_train_times])
    legend = ["DTLearner runtime", "RTLearner runtime"]
    xlabel = "dataset size"
    ylabel = "time to train (s)"
    title = f"Experiment 3, Metric 2 \nDTLearner vs RTLearner Training Time"
    save_path = "images/experiment_3_metric_2.png"
    plot_data(runtimes, legend, xlabel, ylabel, title, save_path, xticks=dataset_sizes)


if __name__ == "__main__":
    np.random.seed(903458910)
    if len(sys.argv) != 2:
        sys.exit(1)
    inf = open(sys.argv[1])

    # ignore the first row and first column (header and date)
    data = np.array(
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
    )

    # because this is time series data - shuffle the rows
    new_idxs = np.random.choice(data.shape[0], size=data.shape[0], replace=False)
    data = data[new_idxs]

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    # run experiment 1
    experiment_1(train_x, train_y, test_x, test_y)

    # run experiment 2
    experiment_2(train_x, train_y, test_x, test_y)

    # run experiment 3, metric 1
    experiment_3_metric_1_RSquared(train_x, train_y, test_x, test_y)

    # run experiment 3, metric 2
    experiment_3_metric_2_traintime(train_x, train_y, test_x, test_y)
