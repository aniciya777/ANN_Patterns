import numpy as np

from .base_initializer import BaseInitializer


class UniformInitializer(BaseInitializer):
    """
    Инициализатор равномерным распределением
    """

    @classmethod
    def initialize(cls, shape: tuple[int, ...], low: float = -1.0, high: float = 1.0, **kwargs) -> np.ndarray:
        """
        Генерация весов нейрона

        :param shape: Размерность вектора весов
        :param low: Минимальное значение
        :param high: Максимальное значение
        :return: Вектор весов нейрона
        """
        return cls._generator.uniform(low, high, shape)
