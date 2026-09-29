"""
Loading the circuit photo and turning it into a pixel-perfect "wall" mask.
"""

import sys
import pygame
import numpy as np


# How to decide which pixels are "wall":
#   WALL_IS_DARK = True  -> pixels DARKER than the threshold are walls
#   WALL_IS_DARK = False -> pixels LIGHTER than the threshold are walls
WALL_IS_DARK = True
BRIGHTNESS_THRESHOLD = 128  # 0 (black) .. 255 (white)


def load_circuit(path: str) -> pygame.Surface:
    """Load the circuit photo from disk.

    Args:
        path: path to the circuit image.

    Returns:
        The raw (not yet display-converted) pygame Surface.
    """
    try:
        image = pygame.image.load(path)
    except FileNotFoundError:
        print(f"\nCould not find '{path}'.")
        print("Run `python make_sample_circuit.py` first, or put your own")
        print(f"circuit photo next to this script and name it '{path}'.\n")
        sys.exit(1)
    return image


def build_wall_mask(image: pygame.Surface):
    """Classify every pixel as wall/path by brightness and return a Mask.

    Returns:
        (wall_mask, debug_surface) — debug_surface is a transparent surface
        you can overlay on screen to visualise exactly what counts as wall.
    """
    arr = pygame.surfarray.array3d(image).astype(np.float32)  # (w, h, 3)
    brightness = arr.mean(axis=2)

    if WALL_IS_DARK:
        is_wall = brightness < BRIGHTNESS_THRESHOLD
    else:
        is_wall = brightness > BRIGHTNESS_THRESHOLD

    alpha = np.where(is_wall, 255, 0).astype(np.uint8)

    mask_surface = pygame.Surface(image.get_size(), pygame.SRCALPHA)
    alpha_view = pygame.surfarray.pixels_alpha(mask_surface)
    alpha_view[:, :] = alpha
    del alpha_view  # release the surface lock

    return pygame.mask.from_surface(mask_surface), mask_surface


def draw_circuit(window: pygame.Surface, circuit_img: pygame.Surface) -> None:
    """Draw the circuit background. Kept as its own function so main.py's
    draw step stays a simple list of draw_xxx() calls."""
    window.blit(circuit_img, (0, 0))