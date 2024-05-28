from abc import ABC

import numpy as np

from ..utils import array


class Loss(ABC):
    """
    Класс функций потерь
    """

    @staticmethod
    def forward(y_pred: array, y_true: array) -> np.float64:
        """
        Алгоритм прямого распространения сигнала

        :param y_pred: Предсказанные значения
        :param y_true: Истинные значения
        :return: Выходное значение функции потерь
        """
        pass

    @staticmethod
    def backward(y_pred: array, y_true: array) -> np.array:
        """
        Алгоритм обратного распространения сигнала

        :param y_pred: Предсказанные значения
        :param y_true: Истинные значения
        :return: Выходное значение функции потерь
        """
        pass
