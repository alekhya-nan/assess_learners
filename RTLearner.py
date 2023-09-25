import numpy as np


class RTLearner(object):
    def __init__(self, leaf_size: int = 1, verbose: bool = False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "anandula3"

    def add_evidence(self, data_x, data_y):
        # have to reshape to work with np.concatenate??
        data_y = data_y.reshape((len(data_y), 1))
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # if we're at a leaf node, then return 0
        if data_x.shape[0] <= self.leaf_size:
            leaf_pred = np.mean(data_y)
            return_leaf = np.array([-1, leaf_pred, -1, -1])
            return return_leaf.reshape((1, 4))

        # if all y values are the same (compare to the first item in the y column), then return leaf node
        # can't use simple equality check bc these are floats
        if np.allclose(data_y.flatten(), data_y[0][0], atol=0.000001):
            return_leaf = np.array([-1, data_y[0][0], -1, -1])
            return return_leaf.reshape((1, 4))

        # get best feature to split on; in RTLearner, it's random
        i = np.random.randint(0, data_x.shape[1])
        split_val = np.median(data_x[:, i])

        # building the left tree
        left_split_condition = data_x[:, i] <= split_val

        # if all the feature i vals are the same, then return a leaf node (causes infinite recursion otherwise: see https://edstem.org/us/courses/43166/discussion/3357228)
        if np.sum(left_split_condition) == len(left_split_condition):
            leaf_pred = np.mean(data_y)
            return_leaf = np.array([-1, leaf_pred, -1, -1])
            return return_leaf.reshape((1, 4))

        left_tree = self.build_tree(
            data_x[left_split_condition], data_y[left_split_condition]
        )

        right_split_condition = data_x[:, i] > split_val
        right_tree = self.build_tree(
            data_x[right_split_condition], data_y[right_split_condition]
        )

        root = np.array([i, split_val, 1, left_tree.shape[0] + 1]).reshape((1, 4))
        concat_tree = np.concatenate([root, left_tree, right_tree], axis=0)

        return concat_tree

    def query(self, points):
        preds = []

        for _, point in enumerate(points):
            pred = self.query_point(point)
            preds.append(pred)

        preds = np.array(preds)

        return preds

    def query_point(self, point):
        current_idx = 0
        ## keep going while the current node is not leaf node
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
