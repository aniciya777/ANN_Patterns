from typing import Union

import numpy as np
from numpy import ndarray

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
        self.__weights = np.zeros(input_size, dtype=np.longdouble)
        self.__bias = np.longdouble(0)
        self.__delta_weights = np.zeros(input_size, dtype=np.longdouble)
        self.__delta_bias = np.longdouble(0)
        self.__last_input: Union[ndarray, None] = None

    def forward(self, x: ndarray) -> np.longdouble:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        self.__last_input = x
        return self.predict(x)

    def backward(self, error: np.longdouble) -> ndarray:
        """
        Алгоритм обратного распространения сигнала

        :param error: Ошибка нейрона
        :return: Ошибка нейрона
        """
        self.__delta_weights = (error * self.__last_input).sum(axis=0)
        self.__delta_bias = error.sum()
        return (error * self.__weights)

    def predict(self, x: ndarray) -> np.longdouble:
        """
        Алгоритм распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        return np.dot(self.__weights, x) + self.__bias

    @property
    def weights(self) -> ndarray:
        """
        Веса нейрона

        :return: Вектор весов нейрона
        """
        return self.__weights

    @property
    def bias(self) -> np.longdouble:
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
        self.__delta_weights = np.zeros(self.__input_size, dtype=np.longdouble)
        self.__delta_bias = np.longdouble(0)

    @property
    def size(self) -> tuple[int, ...]:
        """
        Размер входного вектора

        :return: Размер входного вектора
        """
        return self.__input_size

    @property
    def delta_weights(self) -> ndarray:
        """
        Изменение весов нейрона (геттер)

        :return: Изменение весов нейрона
        """
        return self.__delta_weights

    @property
    def delta_bias(self) -> np.longdouble:
        """
        Изменение смещения нейрона (геттер)

        :return: Изменение смещения нейрона
        """
        return self.__delta_bias

    @delta_weights.setter
    def delta_weights(self, value: ndarray) -> None:
        """
        Изменение весов нейрона (сеттер)

        :param value: Изменение весов нейрона
        """
        self.__delta_weights = value

    @delta_bias.setter
    def delta_bias(self, value: np.longdouble) -> None:
        """
        Изменение смещения нейрона (сеттер)

        :param value: Изменение смещения нейрона
        """
        self.__delta_bias = value

    def update(self) -> None:
        """
        Обновление весов нейрона
        """
        self.__weights -= self.__delta_weights
        self.__bias -= self.__delta_bias
        self.__delta_weights = np.zeros(self.__input_size, dtype=np.longdouble)
        self.__delta_bias = np.longdouble(0)
