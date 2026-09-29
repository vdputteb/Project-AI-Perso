"""
Car classes: physics (accelerate/brake/steer/move) and drawing.

Input handling lives in main.py, collision rules live in game/rules.py —
this module only knows how a car moves and how to draw itself.
"""

import math
import pygame


def rotate_image(window: pygame.Surface, image: pygame.Surface, center: tuple, angle: float) -> None:
    """Draw `image` rotated by `angle` degrees, centered at `center`."""
    rotated_img = pygame.transform.rotate(image, angle)
    new_rect = rotated_img.get_rect(center=center)
    window.blit(rotated_img, new_rect.topleft)


def get_rotated_mask_and_topleft(image: pygame.Surface, center: tuple, angle: float):
    """Build a pixel-perfect mask for the car at its CURRENT rotation,
    plus the on-screen top-left corner it would be drawn at — needed so
    the mask lines up with the wall mask for collision checks."""
    rotated_img = pygame.transform.rotate(image, angle)
    rect = rotated_img.get_rect(center=center)
    return pygame.mask.from_surface(rotated_img), rect.topleft


class AbstractCar:
    """Shared physics for any car (player or, later, AI/opponents)."""

    IMG = None
    START_POS = (0, 0)
    START_ANGLE = 0.0  # degrees; 0 = facing right (east)

    def __init__(self, max_vel: float, rotation_vel: float):
        # Subclasses may already have set self.img (PlayerCar loads its own
        # file per instance); otherwise fall back to the IMG class
        # attribute (AgentCar shares one image across the whole population).
        self.img = getattr(self, "img", self.IMG)
        self.img = pygame.transform.scale(self.img, (50,35))

        self.max_vel = max_vel
        self.rotation_vel = rotation_vel

        self.vel = 0.0
        self.acceleration = 0.15

        self.x, self.y = self.START_POS
        self.angle = self.START_ANGLE

    # ---- steering / throttle -------------------------------------------
    def rotate(self, left: bool = False, right: bool = False) -> None:
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def accelerate(self) -> None:
        self.vel = min(self.vel + self.acceleration, self.max_vel)

    def brake(self) -> None:
        self.vel = max(self.vel - self.acceleration, -self.max_vel / 2)

    def reduce_speed(self) -> None:
        """Friction applied when no throttle key is held."""
        if self.vel > 0:
            self.vel = max(self.vel - self.acceleration / 2, 0)
        elif self.vel < 0:
            self.vel = min(self.vel + self.acceleration / 2, 0)

    def drive(self, steer: float, throttle: float) -> None:
        """Continuous control, used by neural-network-driven cars instead
        of rotate()/accelerate()/brake(). steer and throttle are expected
        in [-1, 1] (e.g. straight out of a tanh output layer)."""
        self.angle += steer * self.rotation_vel

        if throttle >= 0:
            self.vel = min(self.vel + throttle * self.acceleration, self.max_vel)
        else:
            self.vel = max(self.vel + throttle * self.acceleration, -self.max_vel / 2)

    # ---- movement --------------------------------------------------------
    def move(self) -> None:
        radians = math.radians(self.angle)
        self.x += math.cos(radians) * self.vel
        self.y -= math.sin(radians) * self.vel

    # ---- drawing / collision ----------------------------------------------
    def draw(self, window: pygame.Surface) -> None:
        rotate_image(window, self.img, (self.x, self.y), self.angle)

    def get_mask(self):
        """Returns (mask, topleft) for the car at its current position/angle."""
        return get_rotated_mask_and_topleft(self.img, (self.x, self.y), self.angle)

    def reset(self) -> None:
        self.x, self.y = self.START_POS
        self.angle = self.START_ANGLE
        self.vel = 0.0


class PlayerCar(AbstractCar):
    START_POS = (100, 400)
    START_ANGLE = 90.0

    def __init__(self, image_path: str, max_vel: float = 5.0, rotation_vel: float = 4.0):
        # Loaded here (rather than as a class attribute) so it only touches
        # pygame.image once a display exists, and each car can use its own art.
        self.img = pygame.image.load(image_path).convert_alpha()
        super().__init__(max_vel, rotation_vel)