import numpy as np


class DTLearner(object):
    def __init__(self, leaf_size: int, verbose=False):
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
        if np.allclose(data_y.flatten(), data_y[0][0], atol=0.000001):
            return_leaf = np.array([-1, data_y[0][0], -1, -1])
            return return_leaf.reshape((1, 4))

        # determine best feature to split on
        correlation = np.corrcoef(data_x, data_y, rowvar=False)[-1, :-1]
        correlation = np.abs(correlation)
        i = np.argmax(correlation)
        split_val = np.median(data_x[:, i])

        # building the left tree
        left_split_condition = data_x[:, i] <= split_val

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
    pass


if __name__ == "__main__":
    test_code()
