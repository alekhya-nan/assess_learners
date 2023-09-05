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

import numpy as np
from matplotlib import pyplot as plt
from DTLearner import DTLearner
from BagLearner import BagLearner
from RTLearner import RTLearner

def get_RMSE(pred_y, test_y):
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    return rmse

def get_MAE(pred_y, test_y):
    mae = (np.abs(test_y - pred_y)).sum() / test_y.shape[0]

    return mae

def experiment_1(train_x, train_y, test_x, test_y):
    leaf_sizes = [i for i in range(1, 51)]
    train_rmses =  []
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

    legend = ['in sample', 'out of sample']
    plt.clf()
    plt.plot(rmses.T)
    plt.legend(legend)
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title('Experiment 1: Overfitting wrt leaf size in DTLearner')
    plt.grid(visible=True)
    plt.savefig('images/experiment_1.png')


def experiment_2(train_x, train_y, test_x, test_y):
    leaf_sizes = [i for i in range(1, 51)]
    num_bags = 20
    train_rmses =  []
    test_rmses = []

    for leaf_size in leaf_sizes:
        learner = BagLearner(learner=DTLearner, kwargs={'leaf_size': leaf_size}, bags=num_bags)
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
    legend = ['in sample', 'out of sample']

    plt.clf()
    plt.plot(rmses.T)
    plt.legend(legend)
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title(f'Experiment 2: \nOverfitting wrt leaf size in BagLearner (#bags = {num_bags})')
    plt.grid(visible=True)
    plt.savefig('images/experiment_2.png')

def experiment_3_metric_1_MAE(train_x, train_y, test_x, test_y):
    leaf_sizes = [i for i in range(1, 51)]
    rmses = {
        'DT_train_mae': [],
        'DT_test_mae': [],
        'RT_train_mae': [],
        'RT_test_mae': []
    }

    for leaf_size in leaf_sizes:
        # get DTLearner metrics
        dt_learner = DTLearner(leaf_size)
        dt_learner.add_evidence(train_x, train_y)
        
        # train_maes
        pred_train_y = dt_learner.query(train_x)
        train_mae = get_MAE(pred_train_y, train_y)
        rmses['DT_train_mae'].append(train_mae)

        # test maes
        pred_test_y = dt_learner.query(test_x)
        test_mae = get_MAE(pred_test_y, test_y)
        rmses['DT_test_mae'].append(test_mae)

        # get RTLearner metrics
        rt_learner = RTLearner(leaf_size)
        rt_learner.add_evidence(train_x, train_y)

        # train_maes
        pred_train_y = rt_learner.query(train_x)
        train_mae = get_MAE(pred_train_y, train_y)
        rmses['RT_train_mae'].append(train_mae)

        # test maes
        pred_test_y = rt_learner.query(test_x)
        test_mae = get_MAE(pred_test_y, test_y)
        rmses['RT_test_mae'].append(test_mae)
    
    # plot in-sample results
    plt.clf()
    in_sample_data = np.array([rmses['DT_train_mae'], rmses['RT_train_mae']])
    legend = ['DTLearner', 'RTLearner']
    plt.plot(in_sample_data.T)
    plt.legend(legend)
    plt.xlabel('leaf size')
    plt.ylabel('MAE')
    plt.title(f'Experiment 3, Metric 1 \nDTLearner vs RTLearner using Mean Absolute Error (in-sample)')
    plt.grid(visible=True)
    plt.savefig('images/experiment_3_metric_1_insample.png')

    # plot out-of-sample results
    plt.clf()
    out_sample_data = np.array([rmses['DT_test_mae'], rmses['RT_test_mae']])
    legend = ['DTLearner', 'RTLearner']
    plt.plot(out_sample_data.T)
    plt.legend(legend)
    plt.xlabel('leaf size')
    plt.ylabel('MAE')
    plt.title(f'Experiment 3, Metric 1 \nDTLearner vs RTLearner using Mean Absolute Error (out-of-sample)')
    plt.grid(visible=True)
    plt.savefig('images/experiment_3_metric_1_outsample.png')


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
    #experiment_2(train_x, train_y, test_x, test_y)

    # run experiment 3, metric 1
    experiment_3_metric_1_MAE(train_x, train_y, test_x, test_y)
