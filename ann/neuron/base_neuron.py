from abc import ABC, abstractmethod

import numpy as np
import numpy.dtypes
from numpy import array

from ..inizializators import BaseInitializer


class BaseNeuron(ABC):
    """
    Абстрактный класс для нейрона
    """

    @abstractmethod
    def __init__(self, input_size: tuple[int, ...]) -> None:
        """
        Конструктор класса

        :param input_size: Размер входного вектора
        """
        pass

    @abstractmethod
    def forward(self, x: array) -> np.float64:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        pass

    @abstractmethod
    def backward(self, x: array) -> np.float64:
        """
        Алгоритм обратного распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        pass

    @abstractmethod
    def predict(self, x: array) -> np.float64:
        """
        Алгоритм распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        pass

    @property
    @abstractmethod
    def weights(self) -> array:
        """
        Веса нейрона

        :return: Вектор весов нейрона
        """
        pass

    @property
    @abstractmethod
    def bias(self) -> np.float64:
        """
        Смещение нейрона

        :return: Смещение нейрона
        """
        pass

    @abstractmethod
    def initialize(self, initializer: BaseInitializer) -> None:
        """
        Инициализация весов нейрона

        :param initializer: Инициализатор весов нейрона
        """
        pass

    @property
    @abstractmethod
    def size(self) -> tuple[int, ...]:
        """
        Размер вектора весов нейрона

        :return: Размер вектора весов нейрона
        """
        pass

