from BagLearner import BagLearner
from LinRegLearner import LinRegLearner
import numpy as np
class InsaneLearner(object):
    def __init__(self, verbose: bool = False):
        self.num_bags = 20
        self.num_bag_learners = 20
        self.learners = [
            BagLearner(learner=LinRegLearner, kwargs={}, bags=self.num_bags, boost=False, verbose=verbose)
            for _ in range(self.num_bag_learners)
        ]
    def author(self):
        return "anandula3"
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        values = np.array([learner.query(points) for learner in self.learners])
        return values.mean(axis=0)