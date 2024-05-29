"""
Тестирование алгоритма обратного распространения ошибки
в случае линейной функциональной зависимости
"""

import numpy as np

from ann import Network, Trainer
from ann.layers import InputLayer, FullLayer, Layer
from ann.inizializators import UniformInitializer
from ann.optimizers import BackPropagation
from ann.loss import MSE


def test_network():
    SEED = 0
    UniformInitializer().seed(SEED)
    np.random.seed(SEED)

    N = 1000
    X = np.random.rand(N, 2)
    y1 = X[:, 0] + X[:, 1]
    y2 = X[:, 0] - X[:, 1]
    y = np.stack([y1, y2], axis=1)

    nn = Network(
        Layer(cls=InputLayer, size=2),
        Layer(cls=FullLayer, size=10),
        Layer(cls=FullLayer, size=2),
    ).compile()

    optimizer = BackPropagation(learning_rate=0.1)
    trainer = Trainer(
        nn,
        loss=MSE(),
        optimizer=optimizer,
    )
    trainer.fit(X, y, epochs=100, batch_size=8, shuffle=True)
    assert MSE().forward(nn.predict(X), y).mean() < 1e-6
