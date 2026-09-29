"""
Réseau de neurones codé entièrement à la main — aucune librairie de calcul
ou de machine learning (pas de numpy, pas de pytorch...). Seuls `math` et
`random`, qui font partie de la bibliothèque standard de Python, sont
utilisés.

Le réseau apprend la fonction XOR par rétropropagation du gradient
(descente de gradient classique). XOR n'est pas linéairement séparable,
c'est justement pour ça qu'une couche cachée est nécessaire — un exemple
d'école pour comprendre ce qu'un réseau de neurones fait "sous le capot".
"""

import math
import random

# ---------------------------------------------------------------------------
# Hyperparamètres
# ---------------------------------------------------------------------------
N_INPUTS = 2 # nombre d'entrées
N_HIDDEN = 4 # nombres neurones
N_OUTPUTS = 1 # 1 seule sortie
LEARNING_RATE = 0.5
EPOCHS = 1000000

random.seed(42)  # reproductible


def random_weight() -> float:
    return random.uniform(-1, 1)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(already_activated_value: float) -> float:
    """Dérivée de sigmoid, exprimée à partir de sigmoid(x) déjà calculé
    (évite de recalculer exp() une deuxième fois)."""
    y = already_activated_value
    return y * (1 - y)


# ---------------------------------------------------------------------------
class Neuron:
    """Un neurone = une liste de poids (un par entrée) + un biais."""

    def __init__(self, n_inputs: int):
        self.weights = [random_weight() for _ in range(n_inputs)]
        self.bias = random_weight()
        self.output = 0.0  # dernière sortie calculée, réutilisée en backprop
        self.delta = 0.0   # erreur locale, calculée pendant la rétropropagation

    def activate(self, inputs: list) -> float:
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = sigmoid(weighted_sum)
        return self.output


class Layer:
    """Une couche = simplement une liste de neurones indépendants."""

    def __init__(self, n_neurons: int, n_inputs_per_neuron: int):
        self.neurons = [Neuron(n_inputs_per_neuron) for _ in range(n_neurons)]

    def forward(self, inputs: list) -> list:
        return [neuron.activate(inputs) for neuron in self.neurons]


class NeuralNetwork:
    """Un perceptron multicouche : une couche cachée + une couche de sortie."""

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        self.hidden_layer = Layer(n_hidden, n_inputs)
        self.output_layer = Layer(n_outputs, n_hidden)

    def forward(self, inputs: list):
        hidden_outputs = self.hidden_layer.forward(inputs)
        final_outputs = self.output_layer.forward(hidden_outputs)
        return hidden_outputs, final_outputs

    def train_step(self, inputs: list, targets: list, learning_rate: float) -> float:
        hidden_outputs, final_outputs = self.forward(inputs)

        # 1. Erreur de la couche de sortie : (attendu - obtenu) * dérivée
        for neuron, target in zip(self.output_layer.neurons, targets):
            error = target - neuron.output
            neuron.delta = error * sigmoid_derivative(neuron.output)

        # 2. Erreur de la couche cachée : rétropropagée depuis la sortie.
        #    Chaque neurone caché "hérite" de l'erreur des neurones de sortie
        #    auxquels il est connecté, pondérée par le poids de la connexion.
        for i, neuron in enumerate(self.hidden_layer.neurons):
            error = sum(
                out_neuron.weights[i] * out_neuron.delta
                for out_neuron in self.output_layer.neurons
            )
            neuron.delta = error * sigmoid_derivative(neuron.output)

        # 3. Mise à jour des poids de la couche de sortie
        for neuron in self.output_layer.neurons:
            for i in range(len(neuron.weights)):
                neuron.weights[i] += learning_rate * neuron.delta * hidden_outputs[i]
            neuron.bias += learning_rate * neuron.delta

        # 4. Mise à jour des poids de la couche cachée
        for neuron in self.hidden_layer.neurons:
            for i in range(len(neuron.weights)):
                neuron.weights[i] += learning_rate * neuron.delta * inputs[i]
            neuron.bias += learning_rate * neuron.delta

        # Erreur quadratique moyenne, juste pour suivre l'entraînement
        return sum((t - o) ** 2 for t, o in zip(targets, final_outputs)) / len(targets)


def main():
    # Table de vérité du XOR
    dataset = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]

    net = NeuralNetwork(N_INPUTS, N_HIDDEN, N_OUTPUTS)

    for epoch in range(EPOCHS):
        total_error = 0.0
        for inputs, targets in dataset:
            total_error += net.train_step(inputs, targets, LEARNING_RATE)

        if epoch % 500 == 0:
            print(f"époque {epoch:5d} — erreur moyenne : {total_error / len(dataset):.5f}")

    print("\nRésultats finaux :")
    for inputs, targets in dataset:
        _, outputs = net.forward(inputs)
        prediction = round(outputs[0])
        print(f"  entrée {inputs} -> sortie {outputs[0]:.4f} "
              f"(arrondi: {prediction}, attendu: {targets[0]})")


if __name__ == "__main__":
    main()