"""
Тестирование алгоритма обратного распространения ошибки
в случае нелинейной функциональной зависимости
"""

import numpy as np

from ann import Network, Trainer
from ann.layers import InputLayer, FullLayer, Layer
from ann.layers.activations import Sigmoid, LeRU
from ann.inizializators import UniformInitializer
from ann.schedules import StepSchedule
from ann.optimizers import BackPropagation
from ann.loss import MSE


def test_nonlinear():
    SEED = 1
    UniformInitializer().seed(SEED)
    np.random.seed(SEED)

    N = 1000
    X = np.random.rand(N, 2) * 2 - 1
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    y = y.reshape(-1, 1)
    print(X.shape, y.shape)

    nn = Network(
        Layer(cls=InputLayer, size=2),
        Layer(cls=FullLayer, size=10),
        Layer(cls=Sigmoid),
        Layer(cls=FullLayer, size=10),
        Layer(cls=LeRU),
        Layer(cls=FullLayer, size=1),
    ).compile()

    optimizer = BackPropagation()
    trainer = Trainer(
        nn,
        loss=MSE(),
        optimizer=optimizer,
        scheduler=StepSchedule(learning_rate=0.01, step=40, gamma=0.5),
    )
    trainer.fit(X, y, epochs=120, batch_size=1, shuffle=True)
    assert MSE().forward(nn.predict(X), y).mean() < 1e-4
