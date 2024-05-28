from typing import Iterator

import numpy as np

from .base_layer import BaseLayer


class InputLayer(BaseLayer):
    """
    Класс входного слоя
    """

    def __init__(self,
                 input_size: int,
                 size: int,
                 **kwargs) -> None:
        """
        Конструктор класса

        :param input_size: Размер входного вектора (заглушка)
        :param size: Размер входного вектора
        """
        super().__init__(**kwargs)
        self.__size = size
        self.__weights = None

    @property
    def size(self) -> tuple[int, ...]:
        """
        Размер выходного вектора

        :return: Размер выходного вектора
        """
        return self.__size,

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        self.__weights = x
        return x

    def backward(self, y: np.ndarray) -> None:
        """
        Алгоритм обратного распространения сигнала

        :param y: Входной вектор
        """
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм предсказания

        :param x: Входной вектор
        :return: Выходной вектор
        """
        return x

    def __iter__(self) -> Iterator:
        """
        Итератор слоя

        Паттерн "Iterator"
        """
        return iter(self.__weights)
