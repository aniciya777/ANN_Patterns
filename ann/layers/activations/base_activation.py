from typing import override, Iterator

import numpy as np

from ..base_layer import BaseLayer


class BaseActivation(BaseLayer):
    """
    Базовый активационный слой
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
        self._last_input = None

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
        Заглушка для итератора, так как активационный слой не имеет нейронов
        """
        raise StopIteration

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        self._last_input = x
        return self.predict(x)
