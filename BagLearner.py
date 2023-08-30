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
        bag_values = []

        for learner in self.learners:
            value = learner.query(points)
            bag_values.append(value)

        bag_values = np.array(bag_values)
        values = bag_values.mean(axis=0)
        return values

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
