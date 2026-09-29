"""
Algorithme génétique appliqué aux poids (le "génome") des réseaux de
neurones. Ne connaît rien du jeu ni des voitures — il ne manipule que des
listes de nombres (poids) et des scores de fitness.
"""

import random


def crossover(weights_a: list, weights_b: list) -> list:
    """Croisement uniforme : chaque gène vient au hasard d'un des deux parents."""
    return [a if random.random() < 0.5 else b for a, b in zip(weights_a, weights_b)]


def mutate(weights: list, mutation_rate: float = 0.1, mutation_amount: float = 0.3) -> list:
    """Chaque gène a `mutation_rate` de chance de recevoir un petit bruit
    aléatoire. C'est ce qui introduit de la nouveauté génération après
    génération (sans ça, le croisement seul finirait par stagner)."""
    mutated = []
    for w in weights:
        if random.random() < mutation_rate:
            w += random.uniform(-mutation_amount, mutation_amount)
        mutated.append(w)
    return mutated


def select_parent(ranked_population: list):
    """Sélection par tournoi : prend 3 individus au hasard dans la
    population, garde le meilleur des 3. `ranked_population` est une liste
    de (weights, fitness)."""
    contenders = random.sample(ranked_population, k=min(3, len(ranked_population)))
    return max(contenders, key=lambda item: item[1])[0]


def next_generation(population_with_fitness: list, population_size: int,
                     elitism: int = 2) -> list:
    """Produit la prochaine génération de génomes (listes de poids).

    population_with_fitness: liste de (weights, fitness).
    elitism: nombre des meilleurs individus copiés tels quels, sans
    mutation, pour ne jamais perdre le meilleur résultat trouvé.
    """
    ranked = sorted(population_with_fitness, key=lambda item: item[1], reverse=True)

    new_population = [weights for weights, _ in ranked[:elitism]]

    while len(new_population) < population_size:
        parent_a = select_parent(ranked)
        parent_b = select_parent(ranked)
        child = crossover(parent_a, parent_b)
        child = mutate(child)
        new_population.append(child)

    return new_population