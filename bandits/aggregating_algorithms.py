from abc import ABC, abstractmethod
from typing import List

import numpy as np

from bandits.online_models import OnlineModel


class AggregatingAlgorithm(ABC):
    def __init__(self, models: List[OnlineModel]):
        self.models = models
        self.num_models = len(models)
        self.previous_predictions = None

    def get_prediction(self, x: np.ndarray) -> float:
        # x shape is (n_observations,)

        # print(f"predicting for {x=}")
        all_predictions = []
        for model in self.models:
            model_predictions = model.predict(x)
            all_predictions.append(model_predictions)
        all_predictions = np.array(all_predictions).T  # shape=(n_observations, num_models)
        # print(f"{all_predictions=}")
        self.previous_predictions = all_predictions

        final_prediction = self._get_final_prediction(all_predictions)
        return final_prediction

    @abstractmethod
    def _get_final_prediction(self, all_predictions: np.ndarray):
        # all_predictions shape=(n_observations, num_models)
        pass

    def update_estimates(self, x: np.ndarray, y: np.ndarray):
        # x shape=(n_observations,)
        self._update_models(x, y)
        self._update_weights(x, y)

    def _update_models(self, x: np.ndarray, y: np.ndarray):
        # update each model
        for model in self.models:
            model.update(x, y)

    @abstractmethod
    def _update_weights(self, x: np.ndarray, y: np.ndarray):
        # update the weights of each model in aggregate prediction
        pass


class SimpleMeanAggregatingAlgorithm(AggregatingAlgorithm):
    # simply taking the mean of all predictions without updating the weights
    def __init__(self, models: List[OnlineModel]):
        super().__init__(models)

    def _get_final_prediction(self, all_predictions: np.ndarray):
        final_prediction = np.mean(all_predictions, axis=1)
        return final_prediction

    def _update_weights(self, x: np.ndarray, y: np.ndarray):
        pass


class Hedge(AggregatingAlgorithm):
    def __init__(
        self,
        models: List[OnlineModel],
        eta=0.5,
        loss_func=lambda predicted, true: np.abs(predicted - true),
    ):
        super().__init__(models)
        self.eta = eta
        self.weights = np.repeat(1 / self.num_models, self.num_models)
        self.loss_func = loss_func  # the lower - the better
        self.weights_history = []

    def _get_final_prediction(self, all_predictions: np.ndarray):
        final_prediction = all_predictions.dot(self.weights)
        self.weights_history.append(self.weights)
        return final_prediction

    def _update_weights(self, x: np.ndarray, y: np.ndarray):
        losses = np.sum(np.apply_along_axis(lambda x: self.loss_func(x, y), 0, self.previous_predictions), axis=0)
        self.weights = self.weights * np.exp(-self.eta * losses)
        self.weights = self.weights / np.sum(self.weights)
        # print(x.shape, y.shape)
        # print(self.previous_predictions.shape)
        # print()
        # print(f"{losses=}")
