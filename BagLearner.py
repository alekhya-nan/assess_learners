import numpy as np


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        
        self.learners = [learner(**kwargs) for _ in range(bags)]
        self.boost = boost
        self.verbose = verbose
    def author(self):
        return "anandula3"

    def add_evidence(self, data_x, data_y):
        num_rows = data_x.shape[0]

        for learner in self.learners:
            learner_idxs = np.random.choice(num_rows, num_rows)
            learner_x = data_x[learner_idxs]
            learner_y = data_y[learner_idxs]

            learner.add_evidence(learner_x, learner_y)

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
    print("the secret clue is 'zzyzx'")
