"""
Entraîne une population de voitures avec un algorithme génétique pour
faire le MEILLEUR TOUR de circuit (temps au tour le plus court).

Indépendant de main.py (le jeu jouable au clavier) : ce script pilote
automatiquement toute une population de voitures à la fois et fait
évoluer leurs "cerveaux" (réseaux de neurones) génération après
génération, jusqu'à ce que tu arrêtes avec Ctrl+C.

Setup:
    pip install pygame-ce numpy scipy
    python train.py

Le meilleur cerveau trouvé est sauvegardé dans best_brain.json (à chaque
nouveau record, et à l'arrêt) — utilise replay_best.py pour le regarder
rouler seul.

Comment une voiture est notée (voir AI/agent.py) :
  - pas encore de tour bouclé  -> score = distance parcourue sur la piste ;
  - tour bouclé                -> score = uniquement le temps (moins de
                                  frames = mieux), et elle s'arrête.
"""

import json
import math
import os
import sys

import pygame

from AI.agent import AgentCar
from AI.genetic import next_generation, mutate
from AI.neural_network import NeuralNetwork
from game.circuit import load_circuit, draw_circuit, build_wall_mask
from game.rules import car_has_lost
from game.track import build_track, draw_gates


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CIRCUIT_IMAGE_PATH = "circuit2"
CAR_IMAGE_PATH = "game/img/car1.png"
BEST_BRAIN_PATH = f"brains/best_brain_{CIRCUIT_IMAGE_PATH}.json"

POPULATION_SIZE = 60

N_SENSORS = 7          # doit correspondre à len(SENSOR_ANGLES) dans game/sensor.py
N_INPUTS = N_SENSORS + 1   # + vitesse normalisée
N_HIDDEN = 8
N_OUTPUTS = 2           # steer, throttle
LAYER_SIZES = [N_INPUTS, N_HIDDEN, N_OUTPUTS]

CAR_MAX_VEL = 6.0
CAR_ROTATION_VEL = 5.0
SENSOR_MAX_DISTANCE = 220.0

GATE_SPACING = 40           # largeur (px, le long de la piste) d'un checkpoint
FPS = 60                    # sert uniquement à afficher les temps en secondes

MAX_FRAMES_PER_GEN = 4000   # tant qu'aucune voiture n'a bouclé un tour (~67 s)
MAX_STALL_FRAMES = 150      # ~2.5 s sans franchir de porte -> voiture éliminée
LAP_TIME_MARGIN = 1.25      # une fois un tour réussi, la génération est coupée à
                            # record * marge : inutile de laisser les lents finir

RENDER_EVERY_N_GENERATIONS = 1   # 1 = toujours afficher ; augmente pour aller plus vite
DISPLAY_FPS = 60                 # limite d'images/seconde quand une génération est affichée
SAVE_EVERY_N_GENERATIONS = 10    # sauvegarde périodique (en plus de chaque record)
SHOW_CHECKPOINTS = True          # affiche les limites entre checkpoints (orange) et l'arrivée (rouge)


# ---------------------------------------------------------------------------
def spawn_population(weights_list: list) -> list:
    cars = []
    for weights in weights_list:
        brain = NeuralNetwork(LAYER_SIZES)
        brain.set_weights(weights)
        cars.append(AgentCar(CAR_MAX_VEL, CAR_ROTATION_VEL, brain))
    return cars


def step_car(car, wall_mask, track, bounds) -> None:
    """Fait avancer une voiture d'une frame : décision, mouvement, mur,
    progression sur la piste."""
    car.think_and_drive(wall_mask, SENSOR_MAX_DISTANCE)

    prev_pos = (car.x, car.y)
    car.move()

    car_mask, car_topleft = car.get_mask()
    if car_has_lost(wall_mask, car_mask, car_topleft, bounds):
        car.alive = False
        car._update_fitness(track)   # la progression déjà acquise est conservée
        return

    car.update_progress(track, prev_pos, MAX_STALL_FRAMES)


def draw_generation(screen, circuit_img, cars, track, generation, font,
                    best_lap_frames) -> None:
    draw_circuit(screen, circuit_img)

    if SHOW_CHECKPOINTS:
        draw_gates(screen, track)

    for car in cars:
        if car.alive:
            car.draw(screen)

    alive_count = sum(1 for c in cars if c.alive)
    finished = sum(1 for c in cars if c.finished)
    best_fitness = max((c.fitness for c in cars), default=0.0)
    lap = "—" if best_lap_frames is None else f"{best_lap_frames / FPS:.2f}s"
    text = font.render(
        f"génération {generation} — vivantes: {alive_count}/{len(cars)} — "
        f"tours bouclés: {finished} — record: {lap} — score: {best_fitness:.0f}",
        True, (255, 255, 255),
    )
    screen.blit(text, (10, 10))
    pygame.display.flip()


def frame_budget(best_lap_frames) -> int:
    """Durée max d'une génération : large tant qu'aucun tour n'est bouclé,
    puis serrée autour du record (on ne s'intéresse plus qu'à faire mieux)."""
    if best_lap_frames is None:
        return MAX_FRAMES_PER_GEN
    return min(MAX_FRAMES_PER_GEN, int(best_lap_frames * LAP_TIME_MARGIN) + 30)


def run_generation(cars, wall_mask, track, bounds, screen, circuit_img,
                   font, generation: int, render: bool, max_frames: int,
                   best_lap_frames) -> None:
    clock = pygame.time.Clock()

    for _ in range(max_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        any_alive = False
        for car in cars:
            if car.alive:
                step_car(car, wall_mask, track, bounds)
                any_alive = any_alive or car.alive

        if not any_alive:
            break

        if render:
            draw_generation(screen, circuit_img, cars, track, generation, font,
                            best_lap_frames)
            clock.tick(DISPLAY_FPS)  # limité pour rester regardable à l'œil nu
        else:
            clock.tick(0)  # pas de limite de FPS quand on n'affiche pas


# ---------------------------------------------------------------------------
# Sauvegarde / reprise
# ---------------------------------------------------------------------------
def save_best_weights(weights: list, lap_frames=None, path: str = BEST_BRAIN_PATH) -> None:
    """Écriture atomique (fichier temporaire puis remplacement) : un Ctrl+C
    pendant l'écriture ne peut plus corrompre best_brain.json."""
    data = {
        "layer_sizes": LAYER_SIZES,
        "weights": weights,
        "lap_frames": lap_frames,
        "lap_seconds": None if lap_frames is None else round(lap_frames / FPS, 3),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    lap = "pas encore de tour bouclé" if lap_frames is None else f"tour en {lap_frames / FPS:.2f}s"
    print(f"Meilleur cerveau sauvegardé dans {path} ({lap})")


def load_saved_weights(path: str = BEST_BRAIN_PATH):
    """Retourne (poids, lap_frames) sauvegardés, ou (None, None) si le
    fichier n'existe pas, est illisible, ou correspond à une architecture
    différente (LAYER_SIZES changé)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, None
    except (json.JSONDecodeError, OSError) as err:
        print(f"{path} illisible ({err}) — on repart de zéro.")
        return None, None

    if data.get("layer_sizes") != LAYER_SIZES:
        print(f"{path} correspond à une architecture différente "
              f"({data.get('layer_sizes')} au lieu de {LAYER_SIZES}) "
              "— on repart de zéro plutôt que de charger des poids incompatibles.")
        return None, None

    return data["weights"], data.get("lap_frames")


def seed_population_from_weights(best_weights: list, population_size: int) -> list:
    """Première génération construite à partir d'un cerveau déjà entraîné :
    une copie exacte (élitisme, pour ne jamais perdre ce qui a déjà été
    appris) + le reste légèrement mutées pour continuer à explorer autour
    de cette solution plutôt que de repartir de poids aléatoires."""
    population = [list(best_weights)]
    while len(population) < population_size:
        population.append(mutate(list(best_weights), mutation_rate=0.2, mutation_amount=0.4))
    return population


# ---------------------------------------------------------------------------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Entraînement — Circuit Racer IA")

    circuit_img = load_circuit(f"game/img/{CIRCUIT_IMAGE_PATH}.png")
    width, height = circuit_img.get_size()
    screen = pygame.display.set_mode((width, height))
    circuit_img = circuit_img.convert()

    wall_mask, mask_surface = build_wall_mask(circuit_img)
    is_wall = pygame.surfarray.array_alpha(mask_surface) > 0

    # La progression et la ligne d'arrivée sont calculées depuis le point de
    # départ RÉEL des voitures (AgentCar.START_POS / START_ANGLE) : plus de
    # décalage possible entre l'endroit où la voiture apparaît et celui où
    # la piste "commence".
    start = AgentCar.START_POS
    angle = math.radians(AgentCar.START_ANGLE)
    heading = (math.cos(angle), -math.sin(angle))   # y de l'écran vers le bas
    track = build_track(is_wall, start=start, heading=heading, spacing=GATE_SPACING)
    print(f"Piste : {track.n_gates} checkpoints ({track.loop_len:.0f} px par tour), "
          f"{'boucle fermée' if track.closed else 'circuit OUVERT (pas de tour possible)'}.")

    AgentCar.IMG = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()
    font = pygame.font.SysFont(None, 24)

    saved_weights, saved_lap = load_saved_weights(BEST_BRAIN_PATH)
    if saved_weights is not None:
        old = "" if saved_lap is None else f" (tour sauvegardé : {saved_lap / FPS:.2f}s)"
        print(f"Reprise de l'entraînement à partir de {BEST_BRAIN_PATH}{old}.")
        weights_list = seed_population_from_weights(saved_weights, POPULATION_SIZE)
        best_ever_weights = saved_weights
    else:
        print("Aucun cerveau sauvegardé trouvé — nouvel entraînement depuis zéro.")
        weights_list = [NeuralNetwork(LAYER_SIZES).get_weights() for _ in range(POPULATION_SIZE)]
        best_ever_weights = None

    generation = 0
    best_ever_fitness = float("-inf")
    best_lap_frames = None     # record de temps au tour (en frames), ou None

    try:
        while True:
            generation += 1
            cars = spawn_population(weights_list)
            render = (generation % RENDER_EVERY_N_GENERATIONS == 0)

            run_generation(cars, wall_mask, track, (width, height),
                           screen, circuit_img, font, generation, render,
                           frame_budget(best_lap_frames), best_lap_frames)

            scored = [(car.brain.get_weights(), car.fitness) for car in cars]
            best_car = max(cars, key=lambda c: c.fitness)
            best_weights, best_fitness = best_car.brain.get_weights(), best_car.fitness

            # Le score d'un finisseur ne dépend que de son temps, donc comparer
            # les scores entre générations = comparer les temps au tour.
            improved = best_fitness > best_ever_fitness
            if improved:
                best_ever_fitness = best_fitness
                best_ever_weights = best_weights

            if best_car.finished and (best_lap_frames is None
                                      or best_car.lap_frames < best_lap_frames):
                best_lap_frames = best_car.lap_frames
                print(f"*** nouveau record : {best_lap_frames / FPS:.2f}s "
                      f"({best_lap_frames} frames) ***")

            n_fin = sum(1 for c in cars if c.finished)
            lap = "—" if best_lap_frames is None else f"{best_lap_frames / FPS:.2f}s"
            print(f"génération {generation:4d} — checkpoints: {best_car.next_gate:3d}/{track.n_gates} "
                  f"— tours bouclés: {n_fin:2d}/{len(cars)} — record: {lap}")

            # Sauvegarde à chaque amélioration (un Ctrl+C brutal ne perd rien),
            # et périodiquement par sécurité.
            if best_ever_weights is not None and (
                    improved or generation % SAVE_EVERY_N_GENERATIONS == 0):
                save_best_weights(best_ever_weights, best_lap_frames)

            weights_list = next_generation(scored, POPULATION_SIZE)

    except KeyboardInterrupt:
        print("\nEntraînement interrompu.")
        if best_ever_weights is not None:
            save_best_weights(best_ever_weights, best_lap_frames)


if __name__ == "__main__":
    main()