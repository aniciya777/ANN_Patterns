import ann
from .optimizer import Optimizer


class BackPropagation(Optimizer):
    """
    Алгоритм обратного распространения ошибки
    """

    def step(self, trainer: 'ann.Trainer') -> None:
        model = trainer.network
        for layer in reversed(model):
            if isinstance(layer, ann.layers.FullLayer):
                for neuron in layer:
                    neuron.delta_weights *= self._learning_rate
                    neuron.delta_bias *= self._learning_rate
                    neuron.update()
