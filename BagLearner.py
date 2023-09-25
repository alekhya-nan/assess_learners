import numpy as np


class BagLearner(object):
    def __init__(
        self,
        learner,
        kwargs,
        bags: int = 20,
        boost: bool = False,
        verbose: bool = False,
    ):

        self.learners = [learner(**kwargs) for _ in range(bags)]
        self.boost = boost
        self.verbose = verbose

    def author(self):
        return "anandula3"

    def add_evidence(self, data_x, data_y):
        num_rows = data_x.shape[0]

        for learner in self.learners:
            learner_idxs = np.random.choice(a=num_rows, size=num_rows)
            learner_data_x = data_x[learner_idxs]
            learner_data_y = data_y[learner_idxs]

            learner.add_evidence(learner_data_x, learner_data_y)

    def query(self, points):
        learner_preds = [learner.query(points) for learner in self.learners]
        learner_preds = np.array(learner_preds)
        preds = learner_preds.mean(axis=0)

        assert len(preds) == len(points)
        return preds
