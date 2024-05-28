from abc import ABC, abstractmethod

from ..net import Network


class Optimizer(ABC):
    """
    Абстрактный класс оптимизатора
    """
    @abstractmethod
    def step(self, trainer: 'Trainer') -> None:
        pass
