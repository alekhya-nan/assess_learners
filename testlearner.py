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
import matplotlib.pyplot as plt
import numpy as np
from DTLearner import DTLearner

def get_RMSE_corr(pred_y, test_y):
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    corr = np.corrcoef(pred_y, y=test_y)[0,1]

    return rmse, corr

def experiment1(train_x, train_y, test_x, test_y):
    leaf_sizes = [i for i in range(1, 51)]
    metrics = {
        'in_sample_rmse': [],
        'in_sample_corr': [],
        'out_sample_rmse': [],
        'out_sample_corr': [],
    }

    for leaf_size in leaf_sizes:
        learner = DTLearner(leaf_size)
        learner.add_evidence(train_x, train_y)
        
        # in sample values
        pred_train_y = learner.query(train_x)
        rmse, corr = get_RMSE_corr(pred_train_y, train_y)
        metrics['in_sample_rmse'].append(rmse)
        metrics['in_sample_corr'].append(corr)

        pred_y = learner.query(test_x)
        rmse, corr = get_RMSE_corr(pred_y, test_y)
        metrics['out_sample_rmse'].append(rmse)
        metrics['out_sample_corr'].append(corr)

    
    rmses = np.array([metrics['in_sample_rmse'], metrics['out_sample_rmse']])
    print(rmses.shape)
        
    corrs = np.array([metrics['in_sample_corr'], metrics['out_sample_corr']])
    print(corrs.shape)

    plt.plot(rmses.T)
    plt.savefig('images/experiment1_rmse.png')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    inf = open(sys.argv[1])
    # for istanbul, ignore the first row and first column
    data = np.array(
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
    )

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    experiment1(train_x, train_y, test_x, test_y)