"""
Charge best_brain.json (produit par train.py) et fait rouler cette seule
voiture pour observer ce qu'elle a appris.

Usage:
    python replay_best.py
"""

import json
import sys

import pygame

from AI.agent import AgentCar
from AI.neural_network import NeuralNetwork
from game.circuit import load_circuit, draw_circuit, build_wall_mask
from game.rules import car_has_lost
from game.track import generate_checkpoints, draw_checkpoints

CIRCUIT_IMAGE_PATH = "game/img/circuit2.png"
CAR_IMAGE_PATH = "game/img/car1.png"
BEST_BRAIN_PATH = "best_brain.json"

CAR_MAX_VEL = 6.0
CAR_ROTATION_VEL = 5.0
SENSOR_MAX_DISTANCE = 220.0
FPS = 60
SHOW_CHECKPOINTS = True


def load_brain(path: str) -> NeuralNetwork:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\nPas de {path} — lance train.py d'abord (même quelques "
              "générations, puis Ctrl+C pour sauvegarder).\n")
        sys.exit(1)

    brain = NeuralNetwork(data["layer_sizes"])
    brain.set_weights(data["weights"])
    return brain


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Replay — meilleure voiture entraînée")

    circuit_img = load_circuit(CIRCUIT_IMAGE_PATH)
    width, height = circuit_img.get_size()
    screen = pygame.display.set_mode((width, height))
    circuit_img = circuit_img.convert()

    wall_mask, _ = build_wall_mask(circuit_img)
    checkpoints = generate_checkpoints()

    AgentCar.IMG = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()
    brain = load_brain(BEST_BRAIN_PATH)
    car = AgentCar(CAR_MAX_VEL, CAR_ROTATION_VEL, brain)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    car.reset()

        if car.alive:
            car.think_and_drive(wall_mask, SENSOR_MAX_DISTANCE)
            car.move()
            car_mask, car_topleft = car.get_mask()
            if car_has_lost(wall_mask, car_mask, car_topleft, (width, height)):
                car.alive = False
            else:
                car.update_progress(checkpoints)

        draw_circuit(screen, circuit_img)
        if SHOW_CHECKPOINTS:
            draw_checkpoints(screen, checkpoints)
        car.draw(screen)

        status = "EN VIE" if car.alive else "CRASHÉE — R pour relancer"
        text = font.render(
            f"checkpoints: {car.next_checkpoint} — fitness: {car.fitness:.1f} — {status}",
            True, (255, 255, 255),
        )
        screen.blit(text, (10, 10))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()