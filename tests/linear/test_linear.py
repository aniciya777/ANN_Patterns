import numpy as np
import pandas as pd

from ann import Network, Trainer
from ann.layers import InputLayer, FullLayer, Layer
from ann.inizializators import UniformInitializer
from ann.optimizers import BackPropagation
from ann.loss import MSE

SEED = 0
UniformInitializer().seed(SEED)

df = pd.read_csv('data/kc_final.csv')
df = df.drop(['id', 'date', 'zipcode'], axis=1)
X = df.drop('price', axis=1)
y = df['price'].to_numpy().reshape(-1, 1)
print(X.shape, y.shape)

nn = Network(
    Layer(cls=InputLayer, size=18),
    Layer(cls=FullLayer, size=10),
    Layer(cls=FullLayer, size=1)
).compile()
y_calc = nn.predict(X)

optimizer = BackPropagation()
trainer = Trainer(
    nn,
    loss=MSE,
    optimizer=optimizer
)
trainer.fit(X, y, epochs=5, batch_size=32, shuffle=True)


