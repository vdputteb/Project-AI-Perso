"""
Géométrie de la piste d'exemple.

WAYPOINTS sert à la fois à dessiner circuit.png (make_sample_circuit.py)
et à générer les checkpoints utilisés par la fitness de l'algorithme
génétique — pour ne jamais les avoir désynchronisés.

Si tu remplaces circuit.png par ta propre photo, ces checkpoints ne
correspondront plus à ton tracé : il faudra alors redéfinir WAYPOINTS
toi-même (une suite de points, du départ à l'arrivée) en suivant le
centre de ta piste.
"""

WAYPOINTS = [
    (100, 450),
    (100, 150),
    (350, 150),
    (350, 380),
    (600, 380),
    (600, 120),
    (800, 120),
    (800, 500),
    (500, 500),
]
PATH_WIDTH = 90


def generate_checkpoints(spacing: float = 40.0):
    """Points régulièrement espacés le long de WAYPOINTS.

    Retourne une liste de (x, y, rayon). 'rayon' est la distance à
    laquelle le centre d'une voiture doit s'approcher pour que le
    checkpoint soit considéré comme atteint.
    """
    checkpoints = []
    radius = PATH_WIDTH / 2

    for (x1, y1), (x2, y2) in zip(WAYPOINTS, WAYPOINTS[1:]):
        segment_length = math_hypot(x2 - x1, y2 - y1)
        steps = max(1, int(segment_length // spacing))
        for step in range(steps):
            t = step / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            checkpoints.append((x, y, radius))

    checkpoints.append((WAYPOINTS[-1][0], WAYPOINTS[-1][1], radius))
    return checkpoints


def math_hypot(dx: float, dy: float) -> float:
    return (dx * dx + dy * dy) ** 0.5