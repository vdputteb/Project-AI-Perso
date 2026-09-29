"""
Capteurs de distance par lancer de rayons (raycasting), utilisés comme
entrées du réseau de neurones : ce sont les "yeux" de la voiture.
"""

import math

# Angles des capteurs, relatifs au cap de la voiture (0 = devant).
# Un éventail devant + deux capteurs latéraux pour voir venir les murs de côté.
SENSOR_ANGLES = (-90, -45, -20, 0, 20, 45, 90)


def cast_ray(wall_mask, origin: tuple, angle_deg: float, max_distance: float,
             step: float = 4.0) -> float:
    """Avance le long d'un rayon jusqu'à toucher un mur (ou sortir de la
    photo, ou atteindre max_distance). Retourne la distance parcourue."""
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), -math.sin(rad)  # -sin car l'axe y de l'écran pointe vers le bas
    x, y = origin
    mask_width, mask_height = wall_mask.get_size()

    distance = 0.0
    while distance < max_distance:
        x += dx * step
        y += dy * step
        distance += step

        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= mask_width or iy >= mask_height:
            return distance  # sortir de la photo compte comme toucher un mur

        if wall_mask.get_at((ix, iy)):
            return distance

    return max_distance


def read_sensors(car, wall_mask, max_distance: float = 200.0) -> list:
    """Renvoie une liste de distances normalisées (0 = mur collé,
    1 = rien détecté jusqu'à max_distance), une par angle de SENSOR_ANGLES."""
    origin = (car.x, car.y)
    readings = []
    for offset in SENSOR_ANGLES:
        distance = cast_ray(wall_mask, origin, car.angle + offset, max_distance)
        readings.append(distance / max_distance)
    return readings