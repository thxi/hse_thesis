from abc import ABC, abstractmethod

import numpy as np


class OnlineModel(ABC):
    @abstractmethod
    def update_estimates(self, x: np.ndarray, y: np.ndarray):
        # updates the estimates of the model
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class ConstantModel(OnlineModel):
    # always predicts a constant
    def __init__(self, constant):
        self.constant = constant

    def update_estimates(self, x: np.ndarray, y: np.ndarray):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(shape=(x.shape[0],)) + self.constant


class PastObservationsModel(OnlineModel):
    def __init__(self, answer_func, initial_prediction=0):
        # answer func receives a vector of past observations [y1, y2, ..., y_t]
        # and outputs the reponse y_t+1
        self.answer_func = answer_func
        self.y = np.array([])
        self.next_prediction = initial_prediction

    def update_estimates(self, x: np.ndarray, y: np.ndarray):
        self.y = np.hstack([self.y, y])
        self.next_prediction = self.answer_func(self.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.next_prediction, x.shape[0])


class SimpleLinearRegressor(OnlineModel):
    def __init__(
        self,
        intercept=None,
        slope=None,
        x_transform=lambda x: x,
        y_transform=lambda y: y,
        y_inv_transform=lambda y: y,
    ):
        self.dots = np.zeros(5)
        self.intercept = intercept
        self.slope = slope
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.y_inv_transform = y_inv_transform

    def update_estimates(self, x: np.ndarray, y: np.ndarray):
        # update rule stolen from
        # https://scaron.info/blog/simple-linear-regression-with-online-updates.html
        x = self.x_transform(x)
        y = self.y_transform(y)
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

        return self.y_inv_transform(self.intercept + self.slope * self.x_transform(x))


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
