from typing import Union

import numpy as np
from numpy import array

from .base_neuron import BaseNeuron
from ..inizializators import BaseInitializer


class Neuron(BaseNeuron):
    """
    Класс нейрона
    """

    def __init__(self, input_size: tuple[int, ...]) -> None:
        """
        Конструктор класса

        :param input_size: Размер входного вектора
        """
        super().__init__(input_size)
        self.__input_size = input_size
        self.__weights = np.zeros(input_size)
        self.__bias = np.float64(0)
        self.__new_weights = np.zeros(input_size)
        self.__new_bias = np.float64(0)
        self.__last_input: Union[array, None] = None

    def forward(self, x: array) -> np.float64:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        self.__last_input = x
        return self.predict(x)

    def backward(self, error: np.float64) -> np.float64:
        """
        Алгоритм обратного распространения сигнала

        :param error: Ошибка нейрона
        :return: Ошибка нейрона
        """
        pass
        # return error * self.__last_input

    def predict(self, x: array) -> np.float64:
        """
        Алгоритм распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        return np.dot(self.__weights, x) + self.__bias

    @property
    def weights(self) -> array:
        """
        Веса нейрона

        :return: Вектор весов нейрона
        """
        return self.__weights

    @property
    def bias(self) -> np.float64:
        """
        Смещение нейрона

        :return: Смещение нейрона
        """
        return self.__bias

    def initialize(self, initializer: BaseInitializer) -> None:
        """
        Инициализация весов нейрона

        :param initializer: Инициализатор весов нейрона
        """
        self.__weights = initializer.initialize(self.__input_size)
        self.__bias = initializer.initialize((1,))[0]
        self.__new_weights = self.__weights.copy()
        self.__new_bias = self.__bias

    @property
    def size(self) -> tuple[int, ...]:
        """
        Размер входного вектора

        :return: Размер входного вектора
        """
        return self.__input_size
