from typing import Iterator

import numpy as np

from .base_layer import BaseLayer
from ..inizializators import BaseInitializer, UniformInitializer
from ..neuron import Neuron


class FullLayer(BaseLayer):
    """
    Класс полносвязного слоя
    """
    def __init__(self,
                 input_size: tuple[int, ...],
                 size: int,
                 initializer: BaseInitializer = None,
                 **kwargs) -> None:
        """
        Конструктор класса FullLayer

        :param input_size: Размер входного вектора
        :param size: Размер выходного вектора
        :param initializer: Инициализатор весов
        """
        super().__init__(**kwargs)
        if len(input_size) != 1:
            raise ValueError('input_size должен быть одномерным')
        self.__input_size = input_size[0]
        self.__output_size = size
        self.__initializer = initializer or UniformInitializer()
        self.__neurons = [Neuron(self.__input_size) for _ in range(size)]
        for neuron in self.__neurons:
            neuron.initialize(self.__initializer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        return np.array([neuron.forward(x) for neuron in self.__neurons])

    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм обратного распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        result = np.zeros((*x.shape[:-1], self.__input_size), dtype=np.float64)
        for i, neuron in enumerate(self):
            result += neuron.backward(x[:, [i]])
        return result

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходной вектор
        """
        return np.array([neuron.predict(x) for neuron in self.__neurons])

    @property
    def size(self) -> tuple[int, ...]:
        """
        Количество нейронов в слое

        :return: Количество нейронов в слое
        """
        return self.__output_size,

    def __iter__(self) -> Iterator[Neuron]:
        """
        Итератор слоя

        Паттерн "Iterator"
        """
        for neuron in self.__neurons:
            yield neuron
