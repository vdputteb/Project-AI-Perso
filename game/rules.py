"""
Game rules: what counts as losing.

Kept separate from car.py (physics/drawing) and circuit.py (the mask
itself) so the "you lose when..." logic is easy to find and change.
"""


def car_hits_wall(wall_mask, car_mask, car_topleft) -> bool:
    """Pixel-perfect overlap test between the wall mask and the car mask."""
    offset = (int(car_topleft[0]), int(car_topleft[1]))
    return wall_mask.overlap(car_mask, offset) is not None


def car_out_of_bounds(car_mask, car_topleft, bounds) -> bool:
    """True if any part of the car has left the circuit photo entirely."""
    width, height = bounds
    mask_width, mask_height = car_mask.get_size()
    x, y = car_topleft
    return x < 0 or y < 0 or x + mask_width > width or y + mask_height > height


def car_has_lost(wall_mask, car_mask, car_topleft, bounds) -> bool:
    """Single entry point main.py calls each frame."""
    return (
        car_out_of_bounds(car_mask, car_topleft, bounds)
        or car_hits_wall(wall_mask, car_mask, car_topleft)
    )
