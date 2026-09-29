"""
Entraîne une population de voitures avec un algorithme génétique pour
boucler le circuit le plus vite possible.

Indépendant de main.py (le jeu jouable au clavier) : ce script pilote
automatiquement toute une population de voitures à la fois et fait
évoluer leurs "cerveaux" (réseaux de neurones) génération après
génération, jusqu'à ce que tu arrêtes avec Ctrl+C.

Setup:
    pip install pygame-ce numpy
    python make_sample_circuit.py   # si circuit.png n'existe pas encore
    python train.py

Le meilleur cerveau trouvé est sauvegardé régulièrement dans
best_brain.json — utilise replay_best.py pour le regarder rouler seul.
"""

import json
import sys

import pygame

from AI.agent import AgentCar
from AI.genetic import next_generation, mutate
from AI.neural_network import NeuralNetwork
from game import track
from game.circuit import load_circuit, draw_circuit, build_wall_mask
from game.rules import car_has_lost
from game.track import generate_checkpoints, draw_checkpoints


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CIRCUIT_IMAGE_PATH = "game/img/circuit2.png"
CAR_IMAGE_PATH = "game/img/car1.png"
BEST_BRAIN_PATH = "best_brain.json"

POPULATION_SIZE = 60

N_SENSORS = 7          # doit correspondre à len(SENSOR_ANGLES) dans game/sensors.py
N_INPUTS = N_SENSORS + 1   # + vitesse normalisée
N_HIDDEN = 8
N_OUTPUTS = 2           # steer, throttle
LAYER_SIZES = [N_INPUTS, N_HIDDEN, N_OUTPUTS]

CAR_MAX_VEL = 6.0
CAR_ROTATION_VEL = 5.0
SENSOR_MAX_DISTANCE = 220.0

MAX_FRAMES_PER_GEN = 1800   # ~30s à 60 fps : filet de sécurité si tout survit
MAX_STALL_FRAMES = 850      # ~2.5s sans nouveau checkpoint -> voiture éliminée

RENDER_EVERY_N_GENERATIONS = 1   # 1 = toujours afficher ; augmente pour aller plus vite
DISPLAY_FPS = 60                 # limite d'images/seconde quand une génération est affichée
SAVE_EVERY_N_GENERATIONS = 10
SHOW_CHECKPOINTS = True          # affiche les checkpoints (cercles jaunes) par-dessus le circuit


# ---------------------------------------------------------------------------
def spawn_population(weights_list: list) -> list:
    cars = []
    for weights in weights_list:
        brain = NeuralNetwork(LAYER_SIZES)
        brain.set_weights(weights)
        cars.append(AgentCar(CAR_MAX_VEL, CAR_ROTATION_VEL, brain))
    return cars


def step_car(car, wall_mask, checkpoints, bounds) -> None:
    """Fait avancer une voiture d'une frame : décision, mouvement, mise à
    jour de la fitness et de l'état vivant/mort."""
    car.think_and_drive(wall_mask, SENSOR_MAX_DISTANCE)
    car.move()
    car.fitness += car.vel * 0.02  # léger bonus de vitesse, pour départager

    car_mask, car_topleft = car.get_mask()
    if car_has_lost(wall_mask, car_mask, car_topleft, bounds):
        car.alive = False
        return

    car.update_progress(checkpoints, MAX_STALL_FRAMES)


def draw_generation(screen, circuit_img, cars, checkpoints, generation, font) -> None:
    draw_circuit(screen, circuit_img)

    if SHOW_CHECKPOINTS:
        draw_checkpoints(screen, checkpoints)

    for car in cars:
        if car.alive:
            car.draw(screen)

    alive_count = sum(1 for c in cars if c.alive)
    best_fitness = max((c.fitness for c in cars), default=0.0)
    text = font.render(
        f"génération {generation} — vivantes: {alive_count}/{len(cars)} — "
        f"meilleure fitness: {best_fitness:.1f}",
        True, (255, 255, 255),
    )
    screen.blit(text, (10, 10))
    pygame.display.flip()


def run_generation(cars, wall_mask, checkpoints, bounds, screen, circuit_img,
                    font, generation: int, render: bool, max_frames: int) -> None:
    clock = pygame.time.Clock()

    for _ in range(max_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        any_alive = False
        for car in cars:
            if car.alive:
                step_car(car, wall_mask, checkpoints, bounds)
                any_alive = any_alive or car.alive

        if not any_alive:
            break

        if render:
            draw_generation(screen, circuit_img, cars, checkpoints, generation, font)
            clock.tick(DISPLAY_FPS)  # limité pour rester regardable à l'œil nu
        else:
            clock.tick(0)  # pas de limite de FPS quand on n'affiche pas


def save_best_weights(weights: list, path: str = BEST_BRAIN_PATH) -> None:
    with open(path, "w") as f:
        json.dump({"layer_sizes": LAYER_SIZES, "weights": weights}, f)
    print(f"Meilleur cerveau sauvegardé dans {path}")


def load_saved_weights(path: str = BEST_BRAIN_PATH):
    """Retourne les poids sauvegardés, ou None si le fichier n'existe pas
    ou correspond à une architecture différente (LAYER_SIZES changé)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None

    if data.get("layer_sizes") != LAYER_SIZES:
        print(f"{path} correspond à une architecture différente "
              f"({data.get('layer_sizes')} au lieu de {LAYER_SIZES}) "
              "— on repart de zéro plutôt que de charger des poids incompatibles.")
        return None

    return data["weights"]


def seed_population_from_weights(best_weights: list, population_size: int) -> list:
    """Première génération construite à partir d'un cerveau déjà entraîné :
    une copie exacte (élitisme, pour ne jamais perdre ce qui a déjà été
    appris) + le reste légèrement mutées pour continuer à explorer autour
    de cette solution plutôt que de repartir de poids aléatoires."""
    population = [list(best_weights)]
    while len(population) < population_size:
        population.append(mutate(list(best_weights), mutation_rate=0.2, mutation_amount=0.4))
    return population


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Entraînement — Circuit Racer IA")

    circuit_img = load_circuit(CIRCUIT_IMAGE_PATH)
    width, height = circuit_img.get_size()
    screen = pygame.display.set_mode((width, height))
    circuit_img = circuit_img.convert()

    wall_mask, mask_surface = build_wall_mask(circuit_img)
    is_wall = pygame.surfarray.array_alpha(mask_surface) > 0

    checkpoints = track.generate_checkpoints(
        is_wall,
        start=(100, 400),   # position de départ de la voiture
        heading=(0, -1),       # sa direction initiale, ex. (0, -1) = vers le haut
        spacing=40,
    )

    AgentCar.IMG = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()
    font = pygame.font.SysFont(None, 24)

    saved_weights = load_saved_weights(BEST_BRAIN_PATH)
    if saved_weights is not None:
        print(f"Reprise de l'entraînement à partir de {BEST_BRAIN_PATH}.")
        weights_list = seed_population_from_weights(saved_weights, POPULATION_SIZE)
        best_ever_weights = saved_weights
    else:
        print("Aucun cerveau sauvegardé trouvé — nouvel entraînement depuis zéro.")
        weights_list = [NeuralNetwork(LAYER_SIZES).get_weights() for _ in range(POPULATION_SIZE)]
        best_ever_weights = None

    generation = 0
    best_ever_fitness = float("-inf")

    try:
        while True:
            generation += 1
            cars = spawn_population(weights_list)
            render = (generation % RENDER_EVERY_N_GENERATIONS == 0)

            run_generation(cars, wall_mask, checkpoints, (width, height),
                           screen, circuit_img, font, generation, render,
                           MAX_FRAMES_PER_GEN)

            scored = [(car.brain.get_weights(), car.fitness) for car in cars]
            best_weights, best_fitness = max(scored, key=lambda item: item[1])

            if best_fitness > best_ever_fitness:
                best_ever_fitness = best_fitness
                best_ever_weights = best_weights

            print(f"génération {generation:4d} — meilleure fitness: {best_fitness:6.2f} "
                  f"— record: {best_ever_fitness:6.2f}")

            if generation % SAVE_EVERY_N_GENERATIONS == 0 and best_ever_weights:
                save_best_weights(best_ever_weights)

            weights_list = next_generation(scored, POPULATION_SIZE)

    except KeyboardInterrupt:
        print("\nEntraînement interrompu.")
        if best_ever_weights is not None:
            save_best_weights(best_ever_weights)


if __name__ == "__main__":
    main()