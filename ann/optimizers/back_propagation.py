from .optimizer import Optimizer
from ..net import Network


class BackPropagation(Optimizer):
    """
    Алгоритм обратного распространения ошибки
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Конструктор класса

        :param learning_rate: Скорость обучения
        """
        self.__learning_rate = learning_rate

    def step(self, trainer: 'Trainer') -> None:
        pass
