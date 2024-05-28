from abc import ABC, abstractmethod
import logging
from typing import Iterator

import numpy as np


class BaseLayer(ABC):
    """
    Абстрактный класс слоя
    """
    def __init__(self, **kwargs) -> None:
        logging.log(logging.DEBUG, f'Layer {self.__class__.__name__} создан')
        if kwargs:
            logging.log(logging.WARNING, f'Layer {self.__class__.__name__} получил неизвестные параметры: {kwargs}')
            self.__dict__.update(kwargs)

    @property
    @abstractmethod
    def size(self) -> tuple[int, ...]:
        """
        Размерность слоя

        :return: Размерность слоя
        """
        pass

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        pass

    @abstractmethod
    def predict(self, x: np.array) -> np.array:
        """
        Алгоритм распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        pass

    @abstractmethod
    def backward(self, y: np.array) -> None:
        """
        Алгоритм обратного распространения сигнала

        :param y: Вектор для обратного распространения
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """
        Итератор слоя

        Паттерн "Iterator"
        """
        pass
