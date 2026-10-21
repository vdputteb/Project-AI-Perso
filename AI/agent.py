"""
AgentCar : une voiture pilotée par un réseau de neurones plutôt que par le
clavier. Fait le lien entre les capteurs (game/sensor.py), le cerveau
(AI/neural_network.py) et la physique (game/car.py).

Objectif : faire le MEILLEUR TOUR, pas survivre longtemps.
  - tant que la voiture n'a pas bouclé le tour, son score = la distance
    parcourue sur la piste (record atteint, en pixels) ;
  - dès qu'elle franchit la ligne d'arrivée après avoir fait le tour, elle
    s'arrête et son score dépend UNIQUEMENT du temps mis : moins de frames =
    meilleur score ;
  - tout finisseur bat n'importe quel non-finisseur.
"""

from game.car import AbstractCar
from game.sensor import read_sensors

LAP_BONUS = 1000.0         # prime pour avoir bouclé le tour
LAP_TIME_BUDGET = 100_000  # frames ; score finisseur = ... + (LAP_TIME_BUDGET - frames)
MIN_LAP_FRACTION = 0.85    # part du tour à avoir parcourue pour que l'arrivée compte
MAX_REGRESS = 150.0        # px de recul par rapport au record -> demi-tour, éliminée
STALL_STEP = 10.0          # px de progrès minimum pour remettre le compteur d'immobilité à 0


class AgentCar(AbstractCar):
    # Source unique du point de départ : train.py construit la piste (et la
    # ligne d'arrivée) à partir de ces deux valeurs.
    # START_ANGLE : 0 = vers la droite, 90 = vers le haut (y de l'écran inversé).
    START_POS = (100, 450)
    START_ANGLE = 90.0

    IMG = None  # à définir une seule fois avant de créer des instances :
    # AgentCar.IMG = pygame.image.load("car.png").convert_alpha()

    def __init__(self, max_vel: float, rotation_vel: float, brain):
        super().__init__(max_vel, rotation_vel)
        self.brain = brain
        self._init_run_state()

    def _init_run_state(self) -> None:
        self.alive = True
        self.finished = False
        self.fitness = 0.0
        self.best_progress = 0.0      # plus grande distance atteinte (px)
        self.frames = 0               # frames écoulées depuis le départ
        self.lap_frames = None        # temps du tour (en frames) si bouclé
        self.frames_since_progress = 0
        self._stall_ref = 0.0
        self._gates_done = 0

    @property
    def next_gate(self) -> int:
        """Numéro du checkpoint atteint (pour l'affichage uniquement)."""
        return self._gates_done

    # ------------------------------------------------------------------
    def think_and_drive(self, wall_mask, sensor_max_distance: float = 200.0) -> None:
        """Lit les capteurs, les passe au réseau, applique la décision."""
        sensors = read_sensors(self, wall_mask, sensor_max_distance)
        inputs = sensors + [self.vel / self.max_vel]
        steer, throttle = self.brain.forward(inputs)
        self.drive(steer, throttle)

    # ------------------------------------------------------------------
    def update_progress(self, track, prev_pos: tuple, max_stall_frames: int = 150) -> None:
        """À appeler une fois par frame APRÈS move().

        prev_pos : position (x, y) avant le déplacement de cette frame
        (sert à détecter le franchissement de la ligne d'arrivée, même à
        pleine vitesse).
        """
        self.frames += 1
        cur = (self.x, self.y)

        # --- ligne d'arrivée / de départ ---------------------------------
        crossing = track.finish.crossing(prev_pos, cur)
        if crossing == -1:               # franchie à l'envers : contre-sens
            self.alive = False
            self._update_fitness(track)
            return
        if (crossing == 1 and track.closed
                and self.best_progress >= track.loop_len * MIN_LAP_FRACTION):
            self.finished = True         # tour bouclé : on arrête, le chrono est figé
            self.lap_frames = self.frames
            self.alive = False
            self._update_fitness(track)
            return

        # --- progression le long de la piste -----------------------------
        p = track.progress_at(self.x, self.y)
        if p >= 0:
            if p > self.best_progress:
                self.best_progress = p
                self._gates_done = int(p // track.spacing)
            # Reculer de beaucoup par rapport au record = demi-tour.
            if p < self.best_progress - MAX_REGRESS:
                self.alive = False
                self._update_fitness(track)
                return

        if self.best_progress - self._stall_ref >= STALL_STEP:
            self._stall_ref = self.best_progress
            self.frames_since_progress = 0
        else:
            self.frames_since_progress += 1
            if self.frames_since_progress > max_stall_frames:
                self.alive = False

        self._update_fitness(track)

    # ------------------------------------------------------------------
    def _update_fitness(self, track) -> None:
        if self.finished:
            self.fitness = track.loop_len + LAP_BONUS + (LAP_TIME_BUDGET - self.lap_frames)
        else:
            self.fitness = self.best_progress

    def reset(self) -> None:
        super().reset()
        self._init_run_state()