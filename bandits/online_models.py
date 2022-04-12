from abc import ABC, abstractmethod

import numpy as np


class OnlineModel(ABC):
    @abstractmethod
    def update(self, x: np.ndarray, y: np.ndarray):
        # updates the estimates of the model
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class ConstantModel(OnlineModel):
    # always predicts a constant
    def __init__(self, constant):
        self.constant = constant

    def update(self, x: np.ndarray, y: np.ndarray):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(shape=(x.shape[0],)) + self.constant


class SimpleLinearRegressor(OnlineModel):
    def __init__(self, intercept=None, slope=None):
        self.dots = np.zeros(5)
        self.intercept = intercept
        self.slope = slope

    def update(self, x: np.ndarray, y: np.ndarray):
        # update rule stolen from
        # https://scaron.info/blog/simple-linear-regression-with-online-updates.html
        self.dots += np.array(
            [
                x.shape[0],
                x.sum(),
                y.sum(),
                np.dot(x, x),
                np.dot(x, y),
            ]
        )
        size, sum_x, sum_y, sum_xx, sum_xy = self.dots
        det = size * sum_xx - sum_x**2
        if det > 1e-10:  # determinant may be zero initially
            self.intercept = (sum_xx * sum_y - sum_xy * sum_x) / det
            self.slope = (sum_xy * size - sum_x * sum_y) / det

    def predict(self, x: np.ndarray) -> np.ndarray:
        # TODO: maybe throw an exception
        assert self.intercept is not None and self.slope is not None

        return self.intercept + self.slope * x


class OnlinePredictionEnv:
    # a simple environment to test the online models
    def __init__(self, questions, answers):
        super(OnlinePredictionEnv, self).__init__()
        self.questions = questions
        self.answers = answers
        self.question_idx = -1

    def get_context(self):
        self.question_idx += 1
        return self.questions[self.question_idx]

    def get_true_value(self):
        return self.answers[self.question_idx]

    def reset(self):
        return 0
