from typing import override

import numpy as np

from .base_activation import BaseActivation


class LeRU(BaseActivation):
    """
    Активационная функция LeRU
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
        result = x.copy()
        result[result < 0] = 0
        return result

    @override
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм обратного распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        result = x.copy()
        result[self._last_output == 0] = 0
        return result
