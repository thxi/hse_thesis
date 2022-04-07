import numpy as np


class SimpleLinearRegressor:
    def __init__(self):
        self.dots = np.zeros(5)
        self.intercept = None
        self.slope = None

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

    def predict(self, x):
        # TODO: maybe throw an exception
        assert self.intercept is not None and self.slope is not None

        return self.intercept + self.slope * x
