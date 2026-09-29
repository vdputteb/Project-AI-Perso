"""
Réseau de neurones feedforward générique, en pur Python (pas de librairie
de calcul). Contrairement à neural_network_from_scratch.py, il n'a pas
besoin d'apprendre par rétropropagation : ses poids sont optimisés par
l'algorithme génétique (game/genetic.py), donc seule l'inférence
(forward) est nécessaire ici.
"""

import random
import math


def tanh(x: float) -> float:
    return math.tanh(x)


class Neuron:
    def __init__(self, n_inputs: int):
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
        self.bias = random.uniform(-1, 1)

    def activate(self, inputs: list) -> float:
        total = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return tanh(total)


class Layer:
    def __init__(self, n_neurons: int, n_inputs_per_neuron: int):
        self.neurons = [Neuron(n_inputs_per_neuron) for _ in range(n_neurons)]

    def forward(self, inputs: list) -> list:
        return [neuron.activate(inputs) for neuron in self.neurons]


class NeuralNetwork:
    """layer_sizes, ex: [8, 8, 2] -> 8 entrées, 1 couche cachée de 8
    neurones, 2 sorties. Toutes les activations sont en tanh, donc chaque
    sortie est comprise entre -1 et 1 — pratique pour piloter direction et
    accélération directement."""

    def __init__(self, layer_sizes: list):
        self.layer_sizes = list(layer_sizes)
        self.layers = [
            Layer(layer_sizes[i + 1], layer_sizes[i])
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, inputs: list) -> list:
        values = inputs
        for layer in self.layers:
            values = layer.forward(values)
        return values

    # ---- représentation "génome" plat, pour l'algorithme génétique --------
    def get_weights(self) -> list:
        flat = []
        for layer in self.layers:
            for neuron in layer.neurons:
                flat.extend(neuron.weights)
                flat.append(neuron.bias)
        return flat

    def set_weights(self, flat: list) -> None:
        i = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                n = len(neuron.weights)
                neuron.weights = list(flat[i:i + n])
                i += n
                neuron.bias = flat[i]
                i += 1

    def clone(self) -> "NeuralNetwork":
        clone = NeuralNetwork(self.layer_sizes)
        clone.set_weights(self.get_weights())
        return clone