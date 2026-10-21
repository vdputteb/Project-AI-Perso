"""
Progression sur la piste, calculée à partir du masque de murs.

Principe : on mesure, pour CHAQUE pixel de piste, la distance à parcourir
depuis le départ en suivant la piste (distance "géodésique", les murs sont
contournés). La progression d'une voiture = cette distance à l'endroit où
elle se trouve. Un "checkpoint" est simplement une tranche de `spacing`
pixels de cette distance : il s'étend sur toute la largeur de la piste.

Pourquoi pas des portes / cercles posés sur une ligne centrale ?
  - une piste très large, ou séparée en plusieurs couloirs par un mur fin
    (comme sur circuit2.png), n'a pas UNE ligne centrale : une voiture qui
    prend l'autre couloir sauterait des portes et son tour ne serait jamais
    compté ;
  - avec la distance, tous les couloirs fonctionnent, et la progression est
    continue (un score plus fin pour l'algorithme génétique).

Sens unique : une ligne d'arrivée (segment mur-à-mur) est posée juste derrière
la voiture de départ. Elle sert de barrière pour le calcul des distances, et
de ligne de chronométrage : la traverser dans le bon sens après avoir fait
le tour = tour bouclé ; la traverser à l'envers = contre-sens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pygame
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Gate:
    """Segment de droite (ici : la ligne d'arrivée), avec un sens de passage."""
    ax: float; ay: float   # extrémité A (contre un mur)
    bx: float; by: float   # extrémité B (contre l'autre mur)
    cx: float; cy: float   # milieu de la piste à cet endroit
    nx: float; ny: float   # sens de passage (vecteur unitaire)

    @property
    def width(self) -> float:
        return math.hypot(self.bx - self.ax, self.by - self.ay)

    def crossing(self, p0: tuple, p1: tuple) -> int:
        """Le déplacement p0 -> p1 traverse-t-il le segment ?
        +1 = dans le bon sens, -1 = à contre-sens, 0 = non."""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        if dx == 0.0 and dy == 0.0:
            return 0
        ex, ey = self.bx - self.ax, self.by - self.ay
        s0 = ex * (p0[1] - self.ay) - ey * (p0[0] - self.ax)
        s1 = ex * (p1[1] - self.ay) - ey * (p1[0] - self.ax)
        if not ((s0 < 0 <= s1) or (s0 > 0 >= s1)):
            return 0
        t0 = dx * (self.ay - p0[1]) - dy * (self.ax - p0[0])
        t1 = dx * (self.by - p0[1]) - dy * (self.bx - p0[0])
        if t0 * t1 > 0:
            return 0
        return 1 if dx * self.nx + dy * self.ny > 0 else -1


@dataclass
class Track:
    dist: np.ndarray                 # (largeur, hauteur) distance depuis le départ, -1 = inaccessible
    finish: Gate                     # ligne d'arrivée (derrière le départ)
    start: tuple
    loop_len: float                  # longueur d'un tour (px, le long de la piste)
    spacing: float                   # largeur d'un checkpoint (px)
    closed: bool = True              # False si le circuit n'est pas une boucle fermée
    _overlay: object = field(default=None, repr=False)

    @property
    def n_gates(self) -> int:
        """Nombre de checkpoints (tranches) sur un tour."""
        return max(1, int(self.loop_len // self.spacing))

    def progress_at(self, x: float, y: float) -> float:
        """Distance parcourue (px) au point (x, y), ou -1 si hors piste."""
        ix, iy = int(x), int(y)
        w, h = self.dist.shape
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return -1.0
        return float(self.dist[ix, iy])


# ---------------------------------------------------------------------------
# Outils géométriques
# ---------------------------------------------------------------------------
def _hits_wall(is_wall: np.ndarray, x: float, y: float) -> bool:
    ix, iy = math.floor(x), math.floor(y)
    w, h = is_wall.shape
    return ix < 0 or iy < 0 or ix >= w or iy >= h or bool(is_wall[ix, iy])


def _march(is_wall, x, y, dx, dy, max_len, step=0.5):
    """Avance depuis (x, y) jusqu'au premier mur ; renvoie ce point prolongé
    de 1 px dans le mur (aucun interstice ne subsiste)."""
    last = (x, y)
    d = step
    while d <= max_len:
        px, py = x + dx * d, y + dy * d
        if _hits_wall(is_wall, px, py):
            break
        last = (px, py)
        d += step
    return last[0] + dx, last[1] + dy


def _make_gate(is_wall, cx, cy, tx, ty, max_half) -> Gate:
    px, py = -ty, tx
    ax, ay = _march(is_wall, cx, cy, px, py, max_half)
    bx, by = _march(is_wall, cx, cy, -px, -py, max_half)
    return Gate(ax, ay, bx, by, cx, cy, tx, ty)


def _segment_pixels(gate: Gate, back: tuple = (0.0, 0.0), thickness: int = 1):
    """Pixels (entiers) couvrant le segment, épaissi vers `back`."""
    n = max(2, int(gate.width * 2))
    t = np.linspace(0.0, 1.0, n + 1)
    xs = gate.ax + (gate.bx - gate.ax) * t
    ys = gate.ay + (gate.by - gate.ay) * t
    for k in range(thickness):
        yield (np.rint(xs + back[0] * k).astype(int),
               np.rint(ys + back[1] * k).astype(int))


def _geodesic_distance(passable: np.ndarray, source: tuple) -> np.ndarray:
    """Distance (px) le long des pixels `passable` depuis `source`, 8 voisins,
    sans couper les coins. -1 = inatteignable."""
    w, h = passable.shape
    index = np.full((w, h), -1, dtype=np.int32)
    n = int(passable.sum())
    index[passable] = np.arange(n, dtype=np.int32)

    rows, cols, weights = [], [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        x0, x1 = max(0, -dx), w - max(0, dx)
        y0, y1 = max(0, -dy), h - max(0, dy)
        ok = passable[x0:x1, y0:y1] & passable[x0 + dx:x1 + dx, y0 + dy:y1 + dy]
        if dx != 0 and dy != 0:  # diagonale : les deux voisins orthogonaux doivent être libres
            ok &= passable[x0 + dx:x1 + dx, y0:y1]
            ok &= passable[x0:x1, y0 + dy:y1 + dy]
        ia = index[x0:x1, y0:y1][ok]
        ib = index[x0 + dx:x1 + dx, y0 + dy:y1 + dy][ok]
        wgt = np.full(ia.shape, math.hypot(dx, dy), dtype=np.float64)
        rows += [ia, ib]
        cols += [ib, ia]
        weights += [wgt, wgt]

    graph = csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    d = dijkstra(graph, directed=True, indices=int(index[source]))
    out = np.full((w, h), -1.0, dtype=np.float32)
    out[passable] = np.where(np.isfinite(d), d, -1.0)
    return out


# ---------------------------------------------------------------------------
def build_track(
    is_wall: np.ndarray,
    start: tuple,
    heading: tuple,
    spacing: float = 40.0,
    start_line_offset: float = 8.0,
) -> Track:
    """
    Args:
        is_wall: tableau booléen (largeur, hauteur), True = mur
                 (même orientation que pygame.surfarray).
        start:   (x, y) centre de la voiture au départ.
        heading: (dx, dy) direction initiale de la voiture (choisit le sens
                 du circuit ; pas besoin d'être normalisé).
        spacing: largeur (px, le long de la piste) d'un checkpoint.
        start_line_offset: distance entre la voiture au départ et la ligne
                 d'arrivée, située juste derrière elle.
    """
    width, height = is_wall.shape
    free = ~is_wall
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < width and 0 <= sy < height) or not free[sx, sy]:
        raise ValueError(f"Le point de départ {start} n'est pas sur la piste.")

    hx, hy = float(heading[0]), float(heading[1])
    norm = math.hypot(hx, hy) or 1.0
    hx, hy = hx / norm, hy / norm

    clearance = ndimage.distance_transform_edt(free)
    max_half = float(clearance.max()) * 2.5 + 10.0

    # --- ligne d'arrivée / barrière : segment mur-à-mur derrière le départ ---
    bx0, by0 = start[0] - hx * start_line_offset, start[1] - hy * start_line_offset
    if _hits_wall(is_wall, bx0, by0):
        raise ValueError(
            "La ligne d'arrivée tombe dans un mur : le départ est trop près "
            "d'un virage. Réduis start_line_offset ou recule le départ."
        )
    finish = _make_gate(is_wall, bx0, by0, hx, hy, max_half)

    blocked = is_wall.copy()
    thickness = 3
    for ix, iy in _segment_pixels(finish, back=(-hx, -hy), thickness=thickness):
        ok = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
        blocked[ix[ok], iy[ok]] = True
    blocked[sx, sy] = False

    # --- distance le long de la piste depuis le départ ---------------------
    dist = _geodesic_distance(~blocked, (sx, sy))

    # Longueur de la boucle = distance à laquelle on atteint l'arrière de la
    # barrière (la ligne d'arrivée vue de l'autre côté).
    closed = False
    loop_len = float(dist.max())
    behind = []
    for ix, iy in _segment_pixels(finish, back=(-hx, -hy), thickness=1):
        qx = np.rint(ix - hx * (thickness + 1.5)).astype(int)
        qy = np.rint(iy - hy * (thickness + 1.5)).astype(int)
        ok = (qx >= 0) & (qy >= 0) & (qx < width) & (qy < height)
        d = dist[qx[ok], qy[ok]]
        behind.extend(d[d >= 0].tolist())
    if behind:
        closed = True
        loop_len = float(np.median(behind))
    else:
        print("ATTENTION : le circuit ne semble pas être une boucle fermée — "
              "aucun tour ne pourra être compté.")

    return Track(dist=dist, finish=finish, start=(float(start[0]), float(start[1])),
                 loop_len=loop_len, spacing=float(spacing), closed=closed)


# ---------------------------------------------------------------------------
# Affichage (debug)
# ---------------------------------------------------------------------------
def _build_overlay(track: Track, color: tuple) -> pygame.Surface:
    """Surface transparente avec la limite entre chaque checkpoint."""
    d = track.dist
    band = np.where(d >= 0, np.floor(d / track.spacing), -1).astype(np.int32)
    edge = np.zeros(band.shape, dtype=bool)
    edge[:-1, :] |= (band[:-1, :] != band[1:, :]) & (band[:-1, :] >= 0) & (band[1:, :] >= 0)
    edge[:, :-1] |= (band[:, :-1] != band[:, 1:]) & (band[:, :-1] >= 0) & (band[:, 1:] >= 0)
    # on ne trace pas la jonction tout au bout (derrière la barrière)
    edge &= d < track.loop_len - 1.0
    edge = ndimage.binary_dilation(edge) & (d >= 0)

    surf = pygame.Surface(band.shape, pygame.SRCALPHA)
    alpha = pygame.surfarray.pixels_alpha(surf)
    rgb = pygame.surfarray.pixels3d(surf)
    rgb[...] = color
    alpha[...] = np.where(edge, 255, 0).astype(np.uint8)
    del alpha, rgb
    return surf


def draw_gates(window: pygame.Surface, track: Track, color: tuple = (255, 170, 0),
               finish_color: tuple = (255, 0, 0), line_width: int = 3) -> None:
    """Debug : limites entre checkpoints (orange) et ligne d'arrivée (rouge)."""
    if track._overlay is None:
        track._overlay = _build_overlay(track, color)
    window.blit(track._overlay, (0, 0))
    f = track.finish
    pygame.draw.line(window, finish_color, (f.ax, f.ay), (f.bx, f.by), line_width)