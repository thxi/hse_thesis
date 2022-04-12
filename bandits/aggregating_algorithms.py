from abc import ABC, abstractmethod
from typing import List

import numpy as np

from bandits.online_models import OnlineModel


class AggregatingAlgorithm(ABC):
    def __init__(self, models: List[OnlineModel]):
        self.models = models
        self.num_models = len(models)
        self.previous_x = None
        self.previous_predictions = None

    def get_prediction(self, x: float):
        x = np.array([x])
        self.previous_x = x

        all_predictions = []
        for model in self.models:
            model_predictions = model.predict(x)[0]
            all_predictions.append(model_predictions)
        all_predictions = np.array(all_predictions)
        final_prediction = self._get_final_prediction(all_predictions)

        self.previous_predictions = all_predictions
        return final_prediction

    @abstractmethod
    def _get_final_prediction(self, all_predictions: np.ndarray):
        # all_predictions shape is (num_models,)
        pass

    def update_estimates(self, true_value):
        self._update_models(true_value)
        # update each agent
        self._update_weights(true_value)
        pass

    def _update_models(self, true_value):
        # update each model
        for model in self.models:
            model.update(x=self.previous_x, y=np.array([true_value]))

    @abstractmethod
    def _update_weights(self, true_value):
        # update the weights of each model in aggregate prediction
        pass


class SimpleMeanAggregatingAlgorithm(AggregatingAlgorithm):
    # simply taking the mean of all predictions without updating the weights
    def __init__(self, models: List[OnlineModel]):
        super().__init__(models)

    def _get_final_prediction(self, all_predictions: np.ndarray):
        final_prediction = np.mean(all_predictions)
        return final_prediction

    def _update_weights(self, true_value):
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
        final_prediction = np.sum(all_predictions * self.weights)
        self.weights_history.append(self.weights)
        return final_prediction

    def _update_weights(self, true_value):
        # print(self.previous_predictions, true_value)
        losses = self.loss_func(self.previous_predictions, np.repeat(true_value, self.num_models))
        # print(losses)
        self.weights = self.weights * np.exp(-self.eta * losses)
        self.weights = self.weights / np.sum(self.weights)
