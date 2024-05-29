from abc import ABC, abstractmethod

import numpy as np

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
    def forward(self, x: np.ndarray) -> np.longdouble:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.longdouble:
        """
        Алгоритм обратного распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.longdouble:
        """
        Алгоритм распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейрона
        """
        pass

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        Веса нейрона

        :return: Вектор весов нейрона
        """
        pass

    @property
    @abstractmethod
    def bias(self) -> np.longdouble:
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

