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

def get_RMSE(pred_y, test_y):
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    return rmse



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

    rmses = np.array([train_rmses, test_rmses])

    plt.clf()
    plt.plot(rmses.T)
    plt.legend(['in sample', 'out of sample'])
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title('Experiment 1: Overfitting wrt leaf size in DTLearner')
    plt.grid(visible=True)
    plt.savefig('images/experiment_1.png')


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
