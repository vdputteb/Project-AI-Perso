"""
Checkpoints de la piste, générés automatiquement à partir du masque de mur.

Plus de WAYPOINTS : quelle que soit la carte (circuit.png), on déduit la
ligne centrale du chemin libre à partir des pixels "non-mur".

Principe :
  1. distance_transform_edt -> distance au mur le plus proche pour chaque
     pixel libre (le centre de la piste = les plus grandes valeurs).
  2. Une petite barrière est posée derrière la voiture de départ, pour que
     le circuit (même en boucle) ne soit parcouru que dans un sens.
  3. Parcours en largeur (BFS) depuis le départ : distance le long de la piste.
  4. Tous les `spacing` pixels, on prend la tranche de piste correspondante
     et on y place un checkpoint sur le point le plus centré. Son rayon est
     la distance au mur en ce point (largeur locale de la piste).
"""

from collections import deque

import numpy as np
import pygame
from scipy import ndimage


def generate_checkpoints(
    is_wall: np.ndarray,
    start: tuple,
    heading: tuple,
    spacing: int = 40,
    radius_scale: float = 1.0,
    min_radius: float = 12.0,
):
    """
    Args:
        is_wall: tableau booléen (largeur, hauteur), True = mur.
                 (même orientation que pygame.surfarray)
        start:   (x, y) position de départ de la voiture.
        heading: (dx, dy) direction initiale de la voiture (pas besoin
                 d'être normalisé). Sert à choisir le sens du circuit.
        spacing: distance (en pixels le long de la piste) entre checkpoints.
        radius_scale: multiplie le rayon (ex: 0.8 pour être plus strict).
        min_radius:   rayon minimal, pour ne pas avoir de checkpoints
                      impossibles à atteindre dans les passages étroits.

    Returns:
        Liste de (x, y, rayon), du départ vers l'arrivée.
    """
    width, height = is_wall.shape
    free = ~is_wall
    clearance = ndimage.distance_transform_edt(free)

    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < width and 0 <= sy < height) or not free[sx, sy]:
        raise ValueError(f"Le point de départ {start} n'est pas sur la piste.")

    # --- sens unique : barrière derrière la voiture -------------------
    hx, hy = float(heading[0]), float(heading[1])
    norm = (hx * hx + hy * hy) ** 0.5 or 1.0
    hx, hy = hx / norm, hy / norm
    nx, ny = -hy, hx  # perpendiculaire

    c = clearance[sx, sy]
    half_len = c * 1.3
    back = c * 0.8 + 3
    blocked = ~free.copy()
    bx, by = sx - hx * back, sy - hy * back
    for t in np.linspace(-half_len, half_len, max(2, int(half_len * 4))):
        for thick in range(3):
            px = int(round(bx + nx * t - hx * thick))
            py = int(round(by + ny * t - hy * thick))
            if 0 <= px < width and 0 <= py < height:
                blocked[px, py] = True
    blocked[sx, sy] = False

    # --- BFS : distance le long de la piste ---------------------------
    dist = np.full((width, height), -1, dtype=np.int32)
    dist[sx, sy] = 0
    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        d = dist[x, y] + 1
        for ax, ay in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= ax < width and 0 <= ay < height \
                    and not blocked[ax, ay] and dist[ax, ay] < 0:
                dist[ax, ay] = d
                queue.append((ax, ay))

    max_band = int(dist.max()) // spacing
    band_index = np.where(dist >= 0, dist // spacing, -1)

    # --- un checkpoint par tranche ------------------------------------
    checkpoints = []
    for band in range(1, max_band + 1):
        band_mask = band_index == band
        if not band_mask.any():
            continue

        # Si la tranche a plusieurs morceaux (petits éclats contre les murs,
        # embranchements), on garde le plus gros : c'est le vrai chemin.
        labels, n = ndimage.label(band_mask)
        if n > 1:
            sizes = ndimage.sum(band_mask, labels, range(1, n + 1))
            band_mask = labels == (int(np.argmax(sizes)) + 1)

        px, py = np.nonzero(band_mask)
        cl = clearance[px, py]
        if cl.max() < min_radius * 0.75:      # éclat contre un mur : ignoré
            continue
        keep = cl >= cl.max() * 0.85          # pixels les plus centrés
        cx, cy = px[keep], py[keep]
        mx, my = cx.mean(), cy.mean()
        i = int(np.argmin((cx - mx) ** 2 + (cy - my) ** 2))
        x, y = int(cx[i]), int(cy[i])

        radius = max(clearance[x, y] * radius_scale, min_radius)
        checkpoints.append((float(x), float(y), float(radius)))

    return checkpoints


def draw_checkpoints(window: pygame.Surface, checkpoints: list,
                      color: tuple = (255, 210, 0), line_width: int = 2,
                      show_numbers: bool = False, font=None) -> None:
    """Debug overlay: draws every checkpoint as a circle outline, so you
    can see exactly where they are relative to the track. Pass a pygame
    font and show_numbers=True to also label each checkpoint's index."""
    for i, (x, y, radius) in enumerate(checkpoints):
        pygame.draw.circle(window, color, (int(x), int(y)), int(radius), line_width)

        if show_numbers and font is not None:
            label = font.render(str(i), True, color)
            window.blit(label, label.get_rect(center=(int(x), int(y))))