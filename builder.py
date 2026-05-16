"""Lamp lithophane builder matching lithophanemaker.com/Lamp Lithophane.html

Parameter names mirror the website exactly:
  1. Lithophane Parameters
     - outer_diameter, wall_height, min_thickness, max_thickness
     - num_sides, frame_width
     - inner_diameter (top opening)
     - top_thickness, top_height
     - bottom_thickness, bottom_height
  2. Interface Parameters
     - socket_outer_diameter, socket_wall_thickness, socket_height
     - socket_tolerance, lip_height, lip_width
  3. Spoke Parameters
     - spoke_count, spoke_width, spoke_thickness

All geometry is fully vectorised with NumPy (no Python vertex loops).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

MAX_ROWS = 600
MAX_COLS = 600


@dataclass
class BuildParams:
    # ---- 1. Lithophane Parameters ----
    outer_diameter: float = 200.0    # mm  outer diameter of shade
    wall_height:    float = 163.0    # mm  height of lit wall
    min_thickness:  float = 0.8     # mm  thinnest point (brightest)
    max_thickness:  float = 3.0     # mm  thickest point (darkest)
    num_sides:      int   = 4       # number of image panels
    frame_width:    float = 5.0     # mm  solid pillar width between panels
    inner_diameter: float = 0.0     # mm  inner bore at top (0 = auto)
    top_thickness:  float = 5.0     # mm  collar wall thickness
    top_height:     float = 8.0     # mm  collar height
    bottom_thickness: float = 5.0   # mm  base ring wall thickness
    bottom_height:    float = 5.0   # mm  base ring height

    # ---- 2. Interface Parameters ----
    socket_outer_diameter:  float = 27.0   # mm  lamp neck outer diam
    socket_wall_thickness:  float = 3.0    # mm
    socket_height:          float = 25.0   # mm
    socket_tolerance:       float = 0.2    # mm  clearance
    lip_height:             float = 3.0    # mm
    lip_width:              float = 1.5    # mm  inward overhang

    # ---- 3. Spoke Parameters ----
    spoke_count:     int   = 4
    spoke_width:     float = 6.0    # mm
    spoke_thickness: float = 6.0    # mm  vertical height of spoke

    # ---- internal / quality ----
    resolution_mm: float = 0.5     # mesh density mm/vertex
    gamma:         float = 2.2


# ---------------------------------------------------------------------------
# Tiny NumPy helpers
# ---------------------------------------------------------------------------

def _smooth_normals(V: np.ndarray, I: np.ndarray) -> np.ndarray:
    N = np.zeros_like(V)
    v0, v1, v2 = V[I[:, 0]], V[I[:, 1]], V[I[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    np.add.at(N, I[:, 0], fn)
    np.add.at(N, I[:, 1], fn)
    np.add.at(N, I[:, 2], fn)
    mag = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.where(mag < 1e-12, 1.0, mag)


def _quads(base: int, R: int, C: int, flip=False) -> np.ndarray:
    r = np.arange(R - 1, dtype=np.int32)
    c = np.arange(C - 1, dtype=np.int32)
    rr, cc = np.meshgrid(r, c, indexing='ij')
    a = base + rr * C + cc
    b = base + rr * C + cc + 1
    c_ = base + (rr + 1) * C + cc
    d  = base + (rr + 1) * C + cc + 1
    if not flip:
        return np.concatenate([
            np.stack([a, b, c_], -1).reshape(-1, 3),
            np.stack([b, d, c_], -1).reshape(-1, 3)])
    else:
        return np.concatenate([
            np.stack([a, c_, b], -1).reshape(-1, 3),
            np.stack([b, c_, d], -1).reshape(-1, 3)])


def _ring_strip(a: np.ndarray, b: np.ndarray, flip=False) -> np.ndarray:
    n   = len(a)
    ain = np.roll(a, -1); bin_ = np.roll(b, -1)
    if not flip:
        return np.concatenate([
            np.stack([a, b, ain], 1),
            np.stack([ain, b, bin_], 1)])
    else:
        return np.concatenate([
            np.stack([a, ain, b], 1),
            np.stack([ain, bin_, b], 1)])


def _cylinder_rings(offset, r, y_bot, y_top, n, outward=True):
    """Return V(2n,3), I(2n,3) for a closed cylinder band."""
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    Vb = np.stack([r*np.cos(a), np.full(n, y_bot), r*np.sin(a)], 1)
    Vt = np.stack([r*np.cos(a), np.full(n, y_top), r*np.sin(a)], 1)
    V  = np.vstack([Vb, Vt])
    ib = np.arange(n, dtype=np.int32) + offset
    it = np.arange(n, dtype=np.int32) + offset + n
    I  = _ring_strip(ib, it, flip=not outward)
    return V, I


def _annulus(offset, r_in, r_out, y, n, face_up=True):
    a  = np.linspace(0, 2*np.pi, n, endpoint=False)
    Vi = np.stack([r_in *np.cos(a), np.full(n,y), r_in *np.sin(a)], 1)
    Vo = np.stack([r_out*np.cos(a), np.full(n,y), r_out*np.sin(a)], 1)
    V  = np.vstack([Vi, Vo])
    ii = np.arange(n, dtype=np.int32) + offset
    io = np.arange(n, dtype=np.int32) + offset + n
    I  = _ring_strip(ii, io, flip=face_up)
    return V, I


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class LithophaneBuilder:
    def __init__(self, p: BuildParams):
        self.p = p

    # ------------------------------------------------------------------
    def build(self, imgs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build full shade mesh: lit panels + frames + top collar + bottom ring."""
        p  = self.p
        R  = float(p.outer_diameter) / 2.0
        H  = float(p.wall_height)
        NS = p.num_sides
        ts = 2.0 * np.pi / NS

        res   = max(float(p.resolution_mm), 0.1)
        circ  = R * ts
        ncols = int(np.clip(round(circ / res),  8, MAX_COLS))
        nrows = int(np.clip(round(H    / res),  8, MAX_ROWS))

        t_min = float(p.min_thickness)
        t_max = float(p.max_thickness)
        fw    = float(p.frame_width)

        # angular half-width of frame at surface radius
        frame_half = (fw / 2.0) / max(R, 1.0)    # radians

        Vl, Il = [], []
        off = 0

        v_arr = np.linspace(0.0, 1.0, nrows, dtype=np.float64)
        y_arr = H * (1.0 - v_arr)                            # top=H, bot=0
        u_arr = np.linspace(0.0, 1.0, ncols, dtype=np.float64)

        for pi in range(NS):
            arc_s = pi * ts + frame_half
            arc_e = pi * ts + ts - frame_half
            angles = arc_s + u_arr * (arc_e - arc_s)         # (C,)

            if imgs[pi] is not None:
                T = self._img_thickness(imgs[pi], nrows, ncols)
            else:
                T = np.full((nrows, ncols), t_min, dtype=np.float64)

            gray = 1.0 - np.clip(
                (T - t_min) / max(t_max - t_min, 1e-6), 0, 1)

            cos_a = np.cos(angles); sin_a = np.sin(angles)

            # outer surface
            r_out = R + T                                      # (R, C)
            xo = r_out * cos_a
            yo = np.broadcast_to(y_arr[:, None], (nrows, ncols)).copy()
            zo = r_out * sin_a
            Vo = np.stack([xo.ravel(), yo.ravel(), zo.ravel()], 1)
            Co = np.repeat(gray.ravel()[:, None], 3, axis=1)
            nx = np.broadcast_to(cos_a[None, :], (nrows, ncols)).ravel()
            nz = np.broadcast_to(sin_a[None, :], (nrows, ncols)).ravel()
            No = np.stack([nx, np.zeros(nrows*ncols), nz], 1)
            Io = _quads(off, nrows, ncols, flip=False)
            Vl.append(Vo); Il.append(Io); off += len(Vo)

            # inner surface (plain cylinder at R)
            xi = R * cos_a
            zi = R * sin_a
            Vi = np.stack([xi.ravel(), yo.ravel(), zi.ravel()], 1)
            Ci = np.full((nrows*ncols, 3), 0.6)
            Ni = np.stack([-nx, np.zeros(nrows*ncols), -nz], 1)
            Ii = _quads(off, nrows, ncols, flip=True)
            Vl.append(Vi); Il.append(Ii); off += len(Vi)

            # edge caps (thin wall top/bottom)
            o_top = np.arange(off - 2*nrows*ncols,
                              off - 2*nrows*ncols + ncols, dtype=np.int32)
            i_top = np.arange(off - nrows*ncols,
                              off - nrows*ncols + ncols, dtype=np.int32)
            o_bot = np.arange(off - 2*nrows*ncols + (nrows-1)*ncols,
                              off - 2*nrows*ncols + nrows*ncols, dtype=np.int32)
            i_bot = np.arange(off - nrows*ncols + (nrows-1)*ncols,
                              off - nrows*ncols + nrows*ncols, dtype=np.int32)
            Il.append(_ring_strip(o_top, i_top, flip=True))
            Il.append(_ring_strip(o_bot, i_bot, flip=False))

        # ----- frame pillars -----
        n_fr = max(8, int(round(fw / res)))
        for k in range(NS):
            theta_c = (k + 1) * ts            # boundary between panel k and k+1
            tl = theta_c - frame_half
            tr = theta_c + frame_half
            Vf, If = self._frame(off, tl, tr, R, H, t_min,
                                 float(p.frame_width),
                                 float(p.max_thickness), nrows)
            Vl.append(Vf); Il.append(If); off += len(Vf)

        # ----- top collar -----
        if p.top_height > 0 and p.top_thickness > 0:
            n_col = max(ncols * NS, 64)
            r_ci  = float(p.inner_diameter) / 2.0 if p.inner_diameter > 0 \
                    else R - t_min
            r_co  = r_ci + float(p.top_thickness)
            Vt, It = self._collar(off, r_ci, r_co, H,
                                  H + float(p.top_height), n_col)
            Vl.append(Vt); Il.append(It); off += len(Vt)

        # ----- bottom ring -----
        if p.bottom_height > 0 and p.bottom_thickness > 0:
            n_col = max(ncols * NS, 64)
            Vb, Ib = self._base_ring(off, R, float(p.bottom_thickness),
                                     -float(p.bottom_height), 0.0, n_col)
            Vl.append(Vb); Il.append(Ib); off += len(Vb)

        V = np.concatenate(Vl).astype(np.float64)
        I = np.concatenate(Il).astype(np.int32)
        C_all = np.full((len(V), 3), 0.8, dtype=np.float64)
        # copy per-panel colours for lit panels
        col_off = 0
        for pi in range(NS):
            n = nrows * ncols * 2
            if pi < len(imgs) and imgs[pi] is not None:
                T = self._img_thickness(imgs[pi], nrows, ncols)
                gray = 1.0 - np.clip(
                    (T - float(p.min_thickness)) /
                    max(float(p.max_thickness) - float(p.min_thickness), 1e-6), 0, 1)
                g = np.repeat(gray.ravel()[:, None], 3, axis=1)
                C_all[col_off:col_off + nrows*ncols] = g
            col_off += n
        N = _smooth_normals(V, I)
        return V, I, N, C_all

    # ------------------------------------------------------------------
    def build_socket(self) -> Optional[Tuple]:
        """Build socket + spokes as a *separate* mesh.
        Returns (V, I, N, C) or None if socket_height == 0.
        """
        p = self.p
        if p.socket_height <= 0:
            return None

        Vl, Il = [], []
        off = 0

        r_neck = float(p.socket_outer_diameter) / 2.0 + float(p.socket_tolerance)
        r_sock = r_neck + float(p.socket_wall_thickness)
        y_bot  = -float(p.bottom_height) if p.bottom_height > 0 else 0.0
        y_top  = y_bot + float(p.socket_height)
        y_lip  = y_top - float(p.lip_height)
        r_lip  = r_neck - float(p.lip_width)
        n_seg  = 64

        # outer wall of socket
        V, I = _cylinder_rings(off, r_sock, y_bot, y_top, n_seg, outward=True)
        Vl.append(V); Il.append(I); off += len(V)

        # inner wall up to lip
        V, I = _cylinder_rings(off, r_neck, y_bot, y_lip, n_seg, outward=False)
        Vl.append(V); Il.append(I); off += len(V)

        # bottom annulus
        V, I = _annulus(off, r_neck, r_sock, y_bot, n_seg, face_up=False)
        Vl.append(V); Il.append(I); off += len(V)

        # lip step (horizontal face at y_lip from r_lip to r_neck)
        if p.lip_height > 0 and p.lip_width > 0:
            V, I = _annulus(off, r_lip, r_neck, y_lip, n_seg, face_up=False)
            Vl.append(V); Il.append(I); off += len(V)
            # lip inner wall from y_lip to y_top
            V, I = _cylinder_rings(off, r_lip, y_lip, y_top, n_seg, outward=False)
            Vl.append(V); Il.append(I); off += len(V)

        # top annulus
        r_in_top = r_lip if (p.lip_height > 0 and p.lip_width > 0) else r_neck
        V, I = _annulus(off, r_in_top, r_sock, y_top, n_seg, face_up=True)
        Vl.append(V); Il.append(I); off += len(V)

        # ---- spokes ----
        if p.spoke_count > 0:
            R_shade = float(p.outer_diameter) / 2.0
            ys = y_bot
            yt = y_bot + float(p.spoke_thickness)
            ang = np.linspace(0, 2*np.pi, p.spoke_count, endpoint=False)
            hw  = float(p.spoke_width) / 2.0
            for a in ang:
                tx = -np.sin(a); tz = np.cos(a)
                rx =  np.cos(a); rz = np.sin(a)
                def pt(r, s, yy):
                    return [r*rx + s*hw*tx, yy, r*rz + s*hw*tz]
                corners = np.array([
                    pt(r_sock, -1, ys), pt(r_sock, +1, ys),
                    pt(R_shade,-1, ys), pt(R_shade,+1, ys),
                    pt(r_sock, -1, yt), pt(r_sock, +1, yt),
                    pt(R_shade,-1, yt), pt(R_shade,+1, yt),
                ], dtype=np.float64)
                faces = np.array([
                    [4,6,5],[5,6,7],   # top
                    [0,1,2],[1,3,2],   # bottom
                    [0,4,1],[4,5,1],   # hub side
                    [2,3,6],[3,7,6],   # rim side
                    [0,2,4],[4,2,6],   # left
                    [1,5,3],[5,7,3],   # right
                ], dtype=np.int32) + off
                Vl.append(corners); Il.append(faces); off += 8

        V_all = np.concatenate(Vl).astype(np.float64)
        I_all = np.concatenate(Il).astype(np.int32)
        C_all = np.full((len(V_all), 3), 0.65, dtype=np.float64)
        N_all = _smooth_normals(V_all, I_all)
        return V_all, I_all, N_all, C_all

    # ------------------------------------------------------------------
    # Image -> thickness
    # ------------------------------------------------------------------
    def _img_thickness(self, img, nrows, ncols) -> np.ndarray:
        p    = self.p
        data = np.asarray(img.resize((ncols, nrows)), dtype=np.float32) / 255.0
        data = np.power(np.clip(data, 0, 1), 1.0 / max(float(p.gamma), 0.1))
        return (float(p.min_thickness)
                + (1.0 - data) * (float(p.max_thickness) - float(p.min_thickness))
                ).astype(np.float64)

    # ------------------------------------------------------------------
    # Frame pillar (solid vertical strip between panels)
    # ------------------------------------------------------------------
    def _frame(self, offset, theta_l, theta_r, R, H,
               t_min, frame_w, frame_t, nrows):
        v   = np.linspace(0, 1, nrows + 1, dtype=np.float64)
        y   = H * (1.0 - v)
        r_i = R
        r_o = R + max(t_min, frame_t * 0.5)

        # 4 verts per level: left_in, right_in, left_out, right_out
        N = nrows + 1
        V = np.empty((N * 4, 3), np.float64)
        V[0::4] = np.stack([r_i*np.cos(theta_l), y, r_i*np.sin(theta_l)], 1)
        V[1::4] = np.stack([r_i*np.cos(theta_r), y, r_i*np.sin(theta_r)], 1)
        V[2::4] = np.stack([r_o*np.cos(theta_l), y, r_o*np.sin(theta_l)], 1)
        V[3::4] = np.stack([r_o*np.cos(theta_r), y, r_o*np.sin(theta_r)], 1)

        k  = np.arange(nrows, dtype=np.int32)
        b0 = offset + k * 4; b1 = offset + (k+1) * 4
        I  = np.concatenate([
            np.stack([b0+2, b0+3, b1+2], 1), np.stack([b0+3, b1+3, b1+2], 1),  # outer
            np.stack([b0+0, b1+0, b0+1], 1), np.stack([b0+1, b1+0, b1+1], 1),  # inner
            np.stack([b0+0, b0+2, b1+0], 1), np.stack([b0+2, b1+2, b1+0], 1),  # left
            np.stack([b0+1, b1+1, b0+3], 1), np.stack([b1+1, b1+3, b0+3], 1),  # right
        ])
        # caps
        et = offset
        I  = np.vstack([I,
            [[et+0, et+1, et+2], [et+1, et+3, et+2]],          # top cap
            [[offset+(nrows)*4+0, offset+(nrows)*4+2,           # bot cap
              offset+(nrows)*4+1],
             [offset+(nrows)*4+1, offset+(nrows)*4+2,
              offset+(nrows)*4+3]]
        ])
        return V, I.astype(np.int32)

    # ------------------------------------------------------------------
    # Top collar (closed hollow cylinder)
    # ------------------------------------------------------------------
    def _collar(self, offset, r_in, r_out, y_bot, y_top, n):
        Vl, Il = [], []
        off = offset
        V, I = _cylinder_rings(off, r_out, y_bot, y_top, n, outward=True)
        Vl.append(V); Il.append(I); off += len(V)
        V, I = _cylinder_rings(off, r_in,  y_bot, y_top, n, outward=False)
        Vl.append(V); Il.append(I); off += len(V)
        V, I = _annulus(off, r_in, r_out, y_bot, n, face_up=False)
        Vl.append(V); Il.append(I); off += len(V)
        V, I = _annulus(off, r_in, r_out, y_top, n, face_up=True)
        Vl.append(V); Il.append(I); off += len(V)
        return np.vstack(Vl), np.vstack(Il)

    # ------------------------------------------------------------------
    # Bottom base ring
    # ------------------------------------------------------------------
    def _base_ring(self, offset, R_shade, thickness, y_bot, y_top, n):
        r_in  = R_shade
        r_out = R_shade + thickness
        Vl, Il = [], []
        off = offset
        V, I = _cylinder_rings(off, r_out, y_bot, y_top, n, outward=True)
        Vl.append(V); Il.append(I); off += len(V)
        V, I = _cylinder_rings(off, r_in,  y_bot, y_top, n, outward=False)
        Vl.append(V); Il.append(I); off += len(V)
        V, I = _annulus(off, r_in, r_out, y_bot, n, face_up=False)
        Vl.append(V); Il.append(I); off += len(V)
        V, I = _annulus(off, r_in, r_out, y_top, n, face_up=True)
        Vl.append(V); Il.append(I); off += len(V)
        return np.vstack(Vl), np.vstack(Il)
