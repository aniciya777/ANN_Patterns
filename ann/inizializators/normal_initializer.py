import numpy as np

from .base_initializer import BaseInitializer


class NormalInitializer(BaseInitializer):
    """
    Инициализатор нормальным распределением
    """

    @classmethod
    def initialize(cls, shape: tuple[int, ...], scale: float = 1.0, mean: float = 0.0, **kwargs) -> np.ndarray:
        """
        Генерация весов нейрона

        :param shape: Размерность вектора весов
        :param scale: Масштаб нормального распределения
        :param mean: Среднее нормального распределения
        :return: Вектор весов нейрона
        """
        result = cls._generator.standard_normal(shape)
        return result * scale + mean
