from abc import ABC, abstractmethod

from ..net import Network


class Optimizer(ABC):
    """
    Абстрактный класс оптимизатора
    """
    def __init__(self, learning_rate: float = 0.01):
        """
        Конструктор класса

        :param learning_rate: Скорость обучения
        """
        self._learning_rate = learning_rate

    @property
    def learning_rate(self) -> float:
        """
        Скорость обучения

        :return: Скорость обучения
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """
        Установка скорости обучения

        :param value: Скорость обучения
        """
        self._learning_rate = value

    @abstractmethod
    def step(self, trainer: 'Trainer') -> None:
        pass
