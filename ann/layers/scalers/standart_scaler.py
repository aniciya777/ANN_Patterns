from typing import override

import numpy as np

from .base_scaler import BaseScaler


class StandardScaler(BaseScaler):
    """
    Класс масштабирования данных с помощью стандартного отклонения
    """

    def __init__(self,
                 mean: float = 0.0,
                 std: float = 1.0,
                 **kwargs):
        """
        Конструктор класса

        :param min_value:
        :param max_value:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.__mean = mean
        self.__std = std
        self.__shifts = np.zeros(self._input_size)
        self.__scales = np.ones(self._input_size)

    @override
    def _fit_forward(self, x: np.ndarray) -> None:
        """
        Обучение слоя по входному вектору

        :param x: Входной вектор
        """
        self.__shifts = np.mean(x, axis=0)
        self.__scales = np.std(x, axis=0)
        self.__scales[self.__scales == 0] = 1

    @override
    def _fit_backward(self, y: np.ndarray) -> None:
        """
        Обучение слоя по выходному вектору

        :param y: Выходной вектор
        """
        self.__shifts = np.mean(y, axis=0)
        self.__scales = 1 / np.std(y, axis=0)

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Подготовленный входной вектор
        """
        return (x - self.__shifts) / self.__scales

    @override
    def backward(self, y: np.ndarray) -> np.ndarray:
        """
        Алгоритм обратного распространения сигнала

        :param y: Входной вектор
        :return: Подготовленный выходной вектор
        """
        return y * self.__scales + self.__shifts
