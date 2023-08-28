import numpy as np
import pandas as pd

class DTLearner(object):
    def __init__(self, leaf_size: int, verbose=False):
        self.leaf_size = leaf_size
        self.verbose=verbose

    def author(self):
        return "anandula3"

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # if we're at a leaf node, then return 0
        assert data_x.shape[1] == 2
        assert data_y.shape[1] == 1
        if data_x.shape[0] <= self.leaf_size:
            leaf_pred = np.mean(data_y)
            return_leaf = np.array([-1, leaf_pred, -1, -1])
            return return_leaf.reshape((1, 4))
        # if all y values are the same (compare to the first item in the y column), then return leaf node
        if np.all(data_y == data_y[0], axis = 0):
            return_leaf = np.array([-1, data_y[0][0], -1, -1])
            return return_leaf.reshape((1, 4))
        
        # determine best feature to split on
        concat_data = np.concatenate([data_x, data_y], axis=1)
        correlation = np.corrcoef(concat_data, rowvar=False)[-1,:-1]
        correlation = np.abs(correlation)
        i = np.argmax(correlation)

        # splitting value
        split_val = np.median(data_x[:,i])

        # building the left tree
        left_split_condition = data_x[:,i]<=split_val
        left_tree = self.build_tree(data_x[left_split_condition], data_y[left_split_condition])
        # building the right tree
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
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[
            -1
        ]


if __name__ == "__main__":
    arr = pd.read_csv("Data/simple.csv", header=None).to_numpy()
    print("simple shape", arr.shape)
    x = arr[:,:2]
    y = arr[:,-1]
    y = y.reshape((len(y), 1))

    print(x[:5], x.shape)
    print(y[:5], y.shape)

    dtlearner = DTLearner(leaf_size=1)
    dtlearner.add_evidence(x, y)
