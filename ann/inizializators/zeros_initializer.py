import numpy as np

from .base_initializer import BaseInitializer


class ZerosInitializer(BaseInitializer):
    """
    Инициализатор нулями
    """

    @classmethod
    def initialize(cls, shape: tuple[int, ...], **kwargs) -> np.ndarray:
        """
        Генерация весов нейрона

        :param shape: Размерность вектора весов
        :return: Вектор весов нейрона
        """
        return np.zeros(shape)
