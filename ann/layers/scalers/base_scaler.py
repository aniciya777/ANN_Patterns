from abc import abstractmethod
from typing import override, Iterator

import numpy as np

from ..base_layer import BaseLayer


class BaseScaler(BaseLayer):
    """
    Класс базового масштабирования
    """

    def __init__(self,
                 input_size: tuple[int, ...],
                 **kwargs):
        """
        Конструктор класса

        :param input_size: Размер входного вектора
        """
        super().__init__(**kwargs)
        self._input_size = input_size

    @property
    @override
    def size(self) -> tuple[int, ...]:
        """
        Размерность слоя

        :return: Размерность слоя
        """
        return self._input_size

    @override
    def __iter__(self) -> Iterator:
        """
        Итератор слоя

        Паттерн "Iterator"
        """
        return self

    def __next__(self):
        """
        Получить следующий элемент слоя
        Заглушка для итератора, так как скейлер не имеет нейронов
        """
        raise StopIteration

    @abstractmethod
    def _fit_forward(self, x: np.ndarray) -> None:
        """
        Обучение слоя по входному вектору

        :param x: Входной вектор
        """
        pass

    @abstractmethod
    def _fit_backward(self, x: np.ndarray) -> np.ndarray:
        """
        Обучение слоя по выходному вектору

        :param x: Выходной вектор
        """
        pass

    def fit(self, x: np.ndarray, reverse: bool = False) -> None:
        """
        Обучение слоя

        :param x: Входной вектор
        :param reverse: Флаг обратного распространения
        """
        if reverse:
            self._fit_backward(x)
        else:
            self._fit_forward(x)

    @override
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        return self.forward(x)
