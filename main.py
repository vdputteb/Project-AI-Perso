"""
Circuit Racer — pygame-ce

Drive a car around a circuit photo with the arrow keys / WASD. The instant
any part of the car touches a wall (or leaves the photo), you lose.

Controls:
    Up / W        - accelerate
    Down / S      - brake / reverse
    Left / A      - steer left
    Right / D     - steer right
    R             - restart after losing
    Esc           - quit

Setup:
    pip install pygame-ce numpy
    python make_sample_circuit.py   # (optional) creates a test circuit.png
    python make_sample_car.py       # (optional) creates a test car.png
    python main.py
"""

import sys
import pygame

from game.car import PlayerCar
from game.circuit import load_circuit, draw_circuit, build_wall_mask
from game.rules import car_has_lost


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CIRCUIT_IMAGE_PATH = "./game/img/circuit2.png"
CAR_IMAGE_PATH = "./game/img/car1.png"

CAR_MAX_VEL = 7.0
CAR_ROTATION_VEL = 4.0

FPS = 60


# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------
def handle_player_input(car: PlayerCar) -> None:
    """Reads currently pressed keys and updates the car's physics state.
    Keeping this separate from main() makes it trivial to swap in AI input,
    a second player, or a replay system later."""
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        car.rotate(left=True)
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        car.rotate(right=True)

    moving = False
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        car.accelerate()
        moving = True
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        car.brake()
        moving = True

    if not moving:
        car.reduce_speed()


def handle_events(car: PlayerCar) -> tuple:
    """Processes the pygame event queue.

    Returns:
        (keep_running, game_over) — game_over is only ever set to False
        here (on restart); main() sets it to True when a collision happens.
    """
    keep_running = True
    reset_requested = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            keep_running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                keep_running = False
            elif event.key == pygame.K_r:
                reset_requested = True

    if reset_requested:
        car.reset()

    return keep_running, reset_requested


# ---------------------------------------------------------------------------
# DRAWING
# ---------------------------------------------------------------------------
def draw_game_over(window: pygame.Surface, fonts) -> None:
    font_big, font_small = fonts
    width, height = window.get_size()

    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    window.blit(overlay, (0, 0))

    msg = font_big.render("YOU TOUCHED A WALL — GAME OVER", True, (255, 255, 255))
    window.blit(msg, msg.get_rect(center=(width // 2, height // 2 - 20)))

    sub = font_small.render("Press R to restart, Esc to quit", True, (230, 230, 230))
    window.blit(sub, sub.get_rect(center=(width // 2, height // 2 + 30)))


def draw(window: pygame.Surface, circuit_img: pygame.Surface, car: PlayerCar,
         game_over: bool, fonts) -> None:
    draw_circuit(window, circuit_img)
    car.draw(window)

    if game_over:
        draw_game_over(window, fonts)

    pygame.display.flip()


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------
def update(car: PlayerCar, wall_mask, bounds) -> bool:
    """Advances physics one step and reports whether the car has now lost."""
    car.move()
    car_mask, car_topleft = car.get_mask()
    return car_has_lost(wall_mask, car_mask, car_topleft, bounds)


# ---------------------------------------------------------------------------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Circuit Racer")

    circuit_img = load_circuit(CIRCUIT_IMAGE_PATH)
    width, height = circuit_img.get_size()
    screen = pygame.display.set_mode((width, height))
    circuit_img = circuit_img.convert()

    wall_mask, _ = build_wall_mask(circuit_img)
    car = PlayerCar(CAR_IMAGE_PATH, CAR_MAX_VEL, CAR_ROTATION_VEL)

    clock = pygame.time.Clock()
    fonts = (pygame.font.SysFont(None, 56), pygame.font.SysFont(None, 28))

    game_over = False
    running = True

    while running:
        clock.tick(FPS)

        running, was_reset = handle_events(car)
        if was_reset:
            game_over = False

        if not game_over:
            handle_player_input(car)
            game_over = update(car, wall_mask, (width, height))

        draw(screen, circuit_img, car, game_over, fonts)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
