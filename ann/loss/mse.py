import numpy as np

from ..loss.loss import Loss
from ..utils import array, convert_to_array


class MSE(Loss):
    """
    Функция потерь среднеквадратичной ошибки
    """
    @staticmethod
    def forward(y_pred: array, y_true: array) -> np.float64:
        """
        Вычисление функции потерь

        :param y_pred: Предсказанные значения
        :param y_true: Истинные значения
        :return: Значение функции потерь
        """
        y_pred = convert_to_array(y_pred)
        y_true = convert_to_array(y_true)
        return np.mean(np.power(y_pred - y_true, 2))

    @staticmethod
    def backward(y_pred: array, y_true: array) -> np.array:
        """
        Вычисление градиента функции потерь

        :param y_pred: Предсказанные значения
        :param y_true: Истинные значения
        :return: Значение градиента функции потерь
        """
        y_pred = convert_to_array(y_pred)
        y_true = convert_to_array(y_true)
        return 2 * (y_pred - y_true) / y_pred.shape[0]
