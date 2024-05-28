from __future__ import annotations
from typing import List, Iterator

import numpy as np

from ..layers import Layer, BaseLayer
from ..utils import convert_to_array, array


class Network:
    """
    Класс нейронной сети

    Паттерн "Строитель"
    """

    def __init__(self, *layers: Layer):
        self.__commands = layers
        self.__layers: List[BaseLayer] = []

    def compile(self) -> Network:
        """
        Компиляция нейронной сети
        """
        self.__layers.clear()
        input_size = None
        for command in self.__commands:
            new_layer = command.compile(input_size=input_size)
            input_size = new_layer.size
            self.__layers.append(new_layer)
        return self

    def __prepare(self, x: array) -> np.array:
        """
        Подготовка входного вектора

        :param x: Входной вектор
        :return: Подготовленный входной вектор
        """
        x = convert_to_array(x)
        return x.reshape(-1, *self.__layers[0].size)

    def __predict(self, x: array) -> np.array:
        for layer in self.__layers:
            x = layer.forward(x)
        return x

    def predict(self, x: array) -> np.array:
        """
        Предсказание нейронной сети

        :param x: Входной вектор
        :return: Выходное значение нейронной сети
        """
        x = self.__prepare(x)
        return np.array(list(map(self.__predict, x)))

    def __forward(self, x: array) -> np.array:
        for layer in self.__layers:
            x = layer.forward(x)
        return x

    def forward(self, x: array) -> np.array:
        """
        Алгоритм прямого распространения сигнала

        :param x: Входной вектор
        :return: Выходное значение нейронной сети
        """
        x = self.__prepare(x)
        return np.array(list(map(self.__forward, x)))

    def backward(self, y: array) -> None:
        """
        Алгоритм обратного распространения сигнала

        :param y: Выходное значение нейронной сети
        """
        y = convert_to_array(y)
        y = y.reshape(-1, *self.__layers[-1].size)
        for layer in reversed(self.__layers):
            y = layer.backward(y)
        #     y = y.reshape(-1, *self.__layers[-1].size)

    def __iter__(self) -> Iterator[BaseLayer]:
        """
        Итератор по слоям нейронной сети

        Паттерн "Iterator"
        """
        return iter(self.__layers)

    def __reversed__(self) -> Iterator[BaseLayer]:
        """
        Итератор по слоям нейронной сети в обратном порядке

        Паттерн "Iterator"
        """
        return reversed(self.__layers)
