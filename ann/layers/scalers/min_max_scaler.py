from typing import override

import numpy as np

from .base_scaler import BaseScaler


class MinMaxScaler(BaseScaler):
    """
    Класс масштабирования MinMax
    """
    def __init__(self,
                 min_value: np.ndarray | None = None,
                 max_value: np.ndarray | None = None,
                 **kwargs):
        """
        Конструктор класса

        :param min_value:
        :param max_value:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.__min_value = min_value or -np.ones(self._input_size)
        self.__max_value = max_value or np.ones(self._input_size)
        self.__shifts = np.zeros(self._input_size)
        self.__scales = np.ones(self._input_size)

    @override
    def _fit_forward(self, x: np.ndarray) -> None:
        """
        Обучение слоя по входному вектору

        :param x: Входной вектор
        """
        self.__shifts = np.min(x, axis=0)
        self.__scales = np.max(x, axis=0) - self.__shifts
        self.__scales[self.__scales == 0] = 1

    @override
    def _fit_backward(self, y: np.ndarray) -> None:
        """
        Обучение слоя по выходному вектору

        :param y: Выходной вектор
        """
        self.__shifts = self.__min_value
        self.__scales = self.__max_value - self.__min_value
        self.__min_value = np.min(y, axis=0)
        self.__max_value = np.max(y, axis=0)


    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Подготовленный входной вектор
        """
        return (x - self.__shifts) / self.__scales * (self.__max_value - self.__min_value) + self.__min_value

    @override
    def backward(self, y: np.ndarray) -> np.ndarray:
        """
        Алгоритм обратного распространения сигнала

        :param y: Входной вектор
        :return: Подготовленный выходной вектор
        """
        return (y - self.__min_value) / (self.__max_value - self.__min_value) * (self.__scales) + self.__shifts
