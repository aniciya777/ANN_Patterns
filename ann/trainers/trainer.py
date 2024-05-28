import numpy as np

from ..net import Network
from ..loss.loss import Loss
from ..optimizers import Optimizer
from ..utils import convert_to_array, array


class Trainer:
    """
    Класс для тренировки модели
    """
    def __init__(self,
                 network: Network,
                 loss: Loss,
                 optimizer: Optimizer,
                 verbose: bool = True) -> None:
        """
        Конструктор класса

        :param network: Модель
        :param loss: Функция потерь
        :param optimizer: Оптимизатор
        :param verbose: Выводить информацию о процессе обучения
        """
        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self,
            x: array,
            y: array,
            epochs: int,
            test_size: float = 0.2,
            batch_size: int = 32,
            shuffle: bool = True) -> None:
        """
        Обучение модели

        :param x: Вектор признаков
        :param y: Вектор целевых значений
        :param epochs: Количество эпох (максимальное количество итераций)
        :param test_size: Размер тестовой выборки
        :param batch_size: Размер батча
        :param shuffle: Перемешивать данные перед каждой эпохой
        """
        x = convert_to_array(x)
        y = convert_to_array(y)
        x_train, x_test, y_train, y_test = self.__split(x, y, test_size)
        for epoch in range(epochs):
            if shuffle:
                x_train, y_train = self.__shuffle(x_train, y_train)
                x_test, y_test = self.__shuffle(x_test, y_test)

            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                self.network.forward(x_batch)
                self.network.backward(self.loss.backward(self.network.predict(x_batch), y_batch))
                self.optimizer.step(self)

            if self.verbose:
                loss_train = self.loss.forward(self.network.predict(x_train), y_train)
                loss_test = self.loss.forward(self.network.predict(x_test), y_test)
                print(f'Epoch: {epoch + 1}, loss_train: {loss_train}, loss_test: {loss_test}')

    @staticmethod
    def __shuffle(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Перемешивание данных

        :param x: Вектор признаков
        :param y: Вектор целевых значений
        :return: Перемешанные данные
        """
        permutation = np.random.permutation(x.shape[0])
        return x[permutation], y[permutation]

    @staticmethod
    def __split(x: np.ndarray, y: np.ndarray, test_size: float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Разделение данных на тестовую и обучающую выборки

        :param x: Вектор признаков
        :param y: Вектор целевых значений
        :param test_size: Размер тестовой выборки
        :return: Тестовая и обучающая выборки
        """
        test_size = int(test_size * x.shape[0])
        permutation = np.random.permutation(x.shape[0])
        x, y = x[permutation], y[permutation]
        return x[:-test_size], x[-test_size:], y[:-test_size], y[-test_size:]
