import numpy as np
import pandas as pd
import util 
import random

class DTLearner(object):
    def __init__(self, leaf_size: int, verbose=False):
        self.leaf_size = leaf_size
        self.verbose=verbose

    def author(self):
        return "anandula3"

    def add_evidence(self, data_x, data_y):
        data_y = data_y.reshape((len(data_y), 1))
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # if we're at a leaf node, then return 0
        if data_x.shape[0] <= self.leaf_size:
            leaf_pred = np.mean(data_y)
            return_leaf = np.array([-1, leaf_pred, -1, -1])
            return return_leaf.reshape((1, 4))

        # if all y values are the same (compare to the first item in the y column), then return leaf node
        if np.allclose(data_y.flatten(), data_y[0][0], atol=0.000001):
            return_leaf = np.array([-1, data_y[0][0], -1, -1])
            return return_leaf.reshape((1, 4))
        
        # determine best feature to split on
        concat_data = np.concatenate([data_x, data_y], axis=1)
        correlation = np.corrcoef(concat_data, rowvar=False)[-1,:-1]
        correlation = np.abs(correlation)
        i = np.argmax(correlation)
        split_val = np.median(data_x[:,i])

        # building the left tree
        left_split_condition = data_x[:,i]<=split_val

        if np.sum(left_split_condition) == len(left_split_condition):
            leaf_pred = np.mean(data_y)
            return_leaf = np.array([-1, leaf_pred, -1, -1])
            return return_leaf.reshape((1, 4))
        
        left_tree = self.build_tree(data_x[left_split_condition], data_y[left_split_condition])
        
        right_split_condition = data_x[:,i]>split_val
        right_tree = self.build_tree(data_x[right_split_condition], data_y[right_split_condition])

        root = np.array([i, split_val, 1, left_tree.shape[0]+1]).reshape((1, 4))
        concat_tree = np.concatenate([root, left_tree, right_tree], axis=0)

        return concat_tree

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        values = []

        for idx, point in enumerate(points):
            value = self.query_point(point)
            values.append(value)

        values = np.array(values)


        return values
    
    def query_point(self, point):
        current_idx = 0
        ## keep going while the 
        node = self.tree[0]
        while node[0] != -1:
            i = int(node[0])
            split_val = node[1]
            if point[i] <= split_val:
                current_idx += node[2]
            else:
                current_idx += node[3]

            current_idx = int(current_idx)
            node = self.tree[current_idx]

        return node[1]


def test_code():
    fake_seed = 1481090001
    np.random.seed = fake_seed		  
    random.seed = fake_seed	

    datafile = "Istanbul.csv"
    with util.get_learner_data_file(datafile) as f:		  
        alldata = np.genfromtxt(f, delimiter=",")		  
        # Skip the date column and header row if we're working on Istanbul data		  
        if datafile == "Istanbul.csv":		  
            alldata = alldata[1:, 1:]		  
        datasize = alldata.shape[0]		  
        cutoff = int(datasize * 0.6)		  
        permutation = np.random.permutation(alldata.shape[0])		  
        #col_permutation = np.random.permutation(alldata.shape[1] - 1)		  
        train_data = alldata[permutation[:cutoff], :]		  
        # train_x = train_data[:,:-1]		  
        train_x = train_data		  
        train_y = train_data[:, -1]		  
        test_data = alldata[permutation[cutoff:], :]		  
        # test_x = test_data[:,:-1]		  
        test_x = test_data
        test_y = test_data[:, -1]		  
    
    dtlearner = DTLearner(leaf_size=1)
    dtlearner.add_evidence(train_x, train_y)

    preds = dtlearner.query(train_x[:10])
    print(preds, train_y[:10])




if __name__ == "__main__":
    test_code()
    '''
    arr = pd.read_csv("Data/simple.csv", header=None).to_numpy()
    print("simple shape", arr.shape)
    x = arr[:,:2]
    y = arr[:,-1]
    y = y.reshape((len(y), 1))

    print(x[:5], x.shape)
    print(y[:5], y.shape)

    dtlearner = DTLearner(leaf_size=1)
    dtlearner.add_evidence(x, y)


    '''
    '''
    fake_seed = 1481090001
    np.random.seed = fake_seed		  
    random.seed = fake_seed	

    datafile = "Istanbul.csv"
    with util.get_learner_data_file(datafile) as f:		  
        alldata = np.genfromtxt(f, delimiter=",")		  
        # Skip the date column and header row if we're working on Istanbul data		  
        if datafile == "Istanbul.csv":		  
            alldata = alldata[1:, 1:]		  
        datasize = alldata.shape[0]		  
        cutoff = int(datasize * 0.6)		  
        permutation = np.random.permutation(alldata.shape[0])		  
        col_permutation = np.random.permutation(alldata.shape[1] - 1)		  
        train_data = alldata[permutation[:cutoff], :]		  
        # train_x = train_data[:,:-1]		  
        train_x = train_data[:, col_permutation]		  
        train_y = train_data[:, -1]		  
        test_data = alldata[permutation[cutoff:], :]		  
        # test_x = test_data[:,:-1]		  
        test_x = test_data[:, col_permutation]		  
        test_y = test_data[:, -1]		  
        msgs = []

    '''
    pass