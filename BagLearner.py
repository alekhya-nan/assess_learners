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
            learner_idxs = np.random.choice(a=num_rows, size=num_rows)
            learner_x = data_x[learner_idxs]
            learner_y = data_y[learner_idxs]

            learner.add_evidence(learner_x, learner_y)

    def query(self, points):
        bag_preds = []

        for learner in self.learners:
            pred = learner.query(points)
            bag_preds.append(pred)

        bag_preds = np.array(bag_preds)
        preds = bag_preds.mean(axis=0)
        assert len(preds) == len(points)
        return preds
