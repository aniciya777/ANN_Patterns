from typing import override

from .base_schedule import BaseSchedule


class StepSchedule(BaseSchedule):
    """
    Класс шаговой схемы обучения
    """

    @override
    def __init__(self,
                 step: int,
                 gamma: float,
                 **kwargs):
        """
        Конструктор класса

        :param step: Количество эпох до изменения скорости обучения
        :param gamma: Коэффициент уменьшения
        """
        super().__init__(**kwargs)
        self._step = step
        self._gamma = gamma

    @override
    def step(self) -> None:
        """
        Шаг схемы обучения

        :return:
        """
        super().step()
        if self._epoch % self._step == 0:
            self._current_learning_rate *= self._gamma
