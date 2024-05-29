from abc import ABC, abstractmethod

import ann


class BaseSchedule(ABC):
    """
    Абстрактный класс для расписания обучения
    """
    def __init__(self, learning_rate: float) -> None:
        """
        Конструктор класса

        :param learning_rate: Начальная скорость обучения
        """
        self._epoch = 0
        self._initial_learning_rate = learning_rate
        self._current_learning_rate = learning_rate
        self._trainer = None

    def set_trainer(self, trainer: 'ann.Trainer') -> None:
        """
        Сохранение ссылки на обучающий алгоритм

        :param trainer: Обучающий алгоритм
        """
        self._trainer = trainer

    @property
    def learning_rate(self) -> float:
        """
        Скорость обучения
        """
        return self._current_learning_rate

    def start(self) -> None:
        """
        Начало обучения
        """
        self._current_learning_rate = self._initial_learning_rate
        self._epoch = 0

    @abstractmethod
    def step(self) -> None:
        """
        Алгоритм шага обучения
        """
        self._epoch += 1
