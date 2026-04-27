from typing import List, Tuple
import numpy as np


def stitch_rings(indices: List[List[int]],
                 inner_idx: List[int],
                 outer_idx: List[int],
                 outward: bool = True) -> None:
    m = len(inner_idx)
    for j in range(m):
        jn = (j + 1) % m
        a0, a1 = inner_idx[j], inner_idx[jn]
        b0, b1 = outer_idx[j], outer_idx[jn]
        if outward:
            indices.append([a0, b0, a1])
            indices.append([a1, b0, b1])
        else:
            indices.append([a0, a1, b0])
            indices.append([a1, b1, b0])


def stitch_wall(vertices, colors, normals, indices,
               base_xyz, tip_xyz,
               color=(0.6, 0.6, 0.6), up=True) -> None:
    start_base = len(vertices)
    for x, y, z in base_xyz:
        vertices.append([x, y, z])
        colors.append(list(color))
        normals.append([0, 1 if up else -1, 0])
    start_tip = len(vertices)
    for x, y, z in tip_xyz:
        vertices.append([x, y, z])
        colors.append(list(color))
        normals.append([0, 1 if up else -1, 0])
    m = len(base_xyz)
    for j in range(m):
        jn = (j + 1) % m
        a = start_base + j
        b = start_base + jn
        c = start_tip  + j
        d = start_tip  + jn
        if up:
            indices.append([a, b, c])
            indices.append([b, d, c])
        else:
            indices.append([a, c, b])
            indices.append([b, c, d])


def build_ring_xyz_at_radius(radius: float, y: float,
                              ncols: int, num_panels: int,
                              panels_present: List[bool]) -> List[List[float]]:
    """Build a complete-circle ring at the given radius and y height.

    Always emits all num_panels arcs so the ring is a closed, gap-free
    circle.  panels_present is kept as a parameter for API compatibility
    but is no longer used to skip arcs – doing so left angular gaps at
    missing-panel boundaries (exactly where pillars sit), causing holes
    in the bottom brim.
    """
    ring = []
    theta_step = 2.0 * np.pi / num_panels
    for p_idx in range(num_panels):
        for j in range(ncols):
            u     = j / (ncols - 1)
            angle = p_idx * theta_step + u * theta_step
            ring.append([radius * np.cos(angle), y, radius * np.sin(angle)])
    return ring
