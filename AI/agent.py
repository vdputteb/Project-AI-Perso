"""
AgentCar : une voiture pilotée par un réseau de neurones plutôt que par le
clavier. Fait le lien entre les capteurs (game/sensors.py), le cerveau
(game/neural_network.py) et la physique (game/car.py).
"""

from game.car import AbstractCar
from game.sensor import read_sensors


class AgentCar(AbstractCar):
    # Départ et cap alignés sur le premier segment de game/track.py
    # (la piste part vers le haut) — à adapter si tu changes le tracé.
    START_POS = (100, 450)
    START_ANGLE = 90.0

    IMG = None  # à définir une seule fois avant de créer des instances :
    # AgentCar.IMG = pygame.image.load("car.png").convert_alpha()

    def __init__(self, max_vel: float, rotation_vel: float, brain):
        super().__init__(max_vel, rotation_vel)
        self.brain = brain
        self.alive = True
        self.fitness = 0.0
        self.next_checkpoint = 0
        self.frames_since_progress = 0

    def think_and_drive(self, wall_mask, sensor_max_distance: float = 200.0) -> None:
        """Lit les capteurs, les passe au réseau, applique la décision."""
        sensors = read_sensors(self, wall_mask, sensor_max_distance)
        inputs = sensors + [self.vel / self.max_vel]
        steer, throttle = self.brain.forward(inputs)
        self.drive(steer, throttle)

    def update_progress(self, checkpoints: list, max_stall_frames: int = 150) -> None:
        """Récompense la voiture quand elle atteint le prochain checkpoint,
        et la marque comme morte si elle stagne trop longtemps (évite les
        voitures immobiles ou qui tournent en rond indéfiniment)."""
        if self.next_checkpoint < len(checkpoints):
            cx, cy, radius = checkpoints[self.next_checkpoint]
            if (self.x - cx) ** 2 + (self.y - cy) ** 2 <= radius ** 2:
                self.next_checkpoint += 1
                self.fitness += 10.0
                self.frames_since_progress = 0
                return

        self.frames_since_progress += 1
        if self.frames_since_progress > max_stall_frames:
            self.alive = False

    def reset(self) -> None:
        super().reset()
        self.alive = True
        self.fitness = 0.0
        self.next_checkpoint = 0
        self.frames_since_progress = 0