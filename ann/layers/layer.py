from typing import Optional

from .base_layer import BaseLayer


class Layer:
    """
    Класс слоя
    Паттерн "Команда"
    """
    def __init__(self, cls, **kwargs):
        self.__cls = cls
        self.__params = kwargs

    def get_params(self) -> dict:
        """
        Получение параметров слоя

        :return: Параметры слоя
        """
        return self.__params

    def compile(self, input_size: Optional[int]) -> BaseLayer:
        """
        Компиляция слоя

        :return: Компилированный слой
        """
        return self.__cls(input_size=input_size, **self.get_params())
