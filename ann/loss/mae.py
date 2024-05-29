import numpy as np

from ..loss.loss import Loss
from ..utils import array, convert_to_array


class MAE(Loss):
    """
    Функция потерь средней абсолютной ошибки
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
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def backward(y_pred: array, y_true: array) -> np.ndarray:
        """
        Вычисление градиента функции потерь

        :param y_pred: Предсказанные значения
        :param y_true: Истинные значения
        :return: Значение градиента функции потерь
        """
        y_pred = convert_to_array(y_pred)
        y_true = convert_to_array(y_true)
        return (y_pred - y_true) / y_pred.shape[0]
