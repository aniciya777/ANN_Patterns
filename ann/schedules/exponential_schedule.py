from typing import override

from .base_schedule import BaseSchedule


class ExponentialSchedule(BaseSchedule):
    """
    Класс экспоненциальной схемы обучения
    """

    @override
    def __init__(self,
                 gamma: float,
                 **kwargs):
        """
        Конструктор класса

        :param gamma: Коэффициент уменьшения
        """
        super().__init__(**kwargs)
        self._gamma = gamma

    @override
    def step(self) -> None:
        """
        Шаг схемы обучения

        :return:
        """
        self._current_learning_rate *= self._gamma
    