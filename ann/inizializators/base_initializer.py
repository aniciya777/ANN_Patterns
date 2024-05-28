from abc import ABC, abstractmethod

import numpy as np


class BaseInitializer(ABC):
    """
    Базовый класс инициализатора

    Паттерн "Стратегия"
    """

    __instance = None
    _generator = np.random.default_rng()

    def __new__(cls, *args, **kwargs):
        """
        Проверка на существование экземпляра инициализатора
        Реализация шаблона Singleton

        :return: Экземпляр инициализатора
        """
        if cls.__instance is None:
            cls.__instance = super(BaseInitializer, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    @classmethod
    def seed(cls, seed: int) -> None:
        """
        Задание начального состояния генератора случайных чисел

        :param seed: Начальное состояние генератора случайных чисел
        """
        cls._generator = np.random.default_rng(seed)

    @classmethod
    @abstractmethod
    def initialize(cls, shape: tuple[int, ...], **kwargs) -> np.ndarray:
        """
        Генерация весов нейрона
        
        :param shape: Размерность вектора весов
        :return: Вектор весов нейрона
        """
        pass
