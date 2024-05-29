from typing import override

import numpy as np

from .base_activation import BaseActivation


class Sigmoid(BaseActivation):
    """
    Активационная функция сигмоида
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_output = None

    @override
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        self._last_output = 1 / (1 + np.exp(-x))
        return self._last_output

    @override
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм обратного распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        return self._last_output * (1 - self._last_output) * x
