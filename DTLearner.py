import numpy as np
import pandas as pd

class DTLearner(object):
    def __init__(self, verbose=False):
        self.verbose=verbose

    def author(self):
        return "anandula3"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        concat_data = np.concatenate([data_x, data_y], axis=1)
        print(concat_data.shape)
        print(self.build_tree(concat_data))

    def build_tree(self, data):
        # if we're at a leaf node, then return 0
        if data.shape[0] == 1:
            return_leaf = np.array([-1, data[0][0], -1, -1])
            return return_leaf.reshape((1, 4))
        # if all y values are the same (compare to the first item in the y column), then return leaf node
        if np.all(data[0,-1] == data[:,-1], axis = 0):
            return_leaf = np.array([-1, data[0][0], -1, -1])
            return return_leaf.reshape((1, 4))
        
        # determine best feature to split on
        i = 0

        
        split_val = np.median(data[:,i])
        left_tree = self.build_tree(data[data[:,i]<=split_val])
        right_tree = self.build_tree(data[data[:,i]>split_val])

        root = np.array([i, split_val, 1, left_tree.shape[0]+1])
        root = root.reshape((1, 4))
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

    dtlearner = DTLearner()
    dtlearner.add_evidence(x, y)
