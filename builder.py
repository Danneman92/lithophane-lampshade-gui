"""LithophaneMaker-parity lamp builder  —  fully vectorised (no Python vertex loops)."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


# Hard cap to keep generation fast regardless of resolution setting.
# 500x500 per panel = 250k verts per panel = still very fine at 163mm height.
MAX_ROWS = 500
MAX_COLS = 500


@dataclass
class BuildParams:
    # --- shape ---
    height: float       = 163.0
    top_diam: float     = 170.0
    bottom_diam: float  = 200.0

    # --- lithophane quality ---
    min_thickness: float  = 0.8
    max_thickness: float  = 3.0
    resolution_mm: float  = 0.5    # mm per pixel
    gamma: float          = 2.2
    contrast: float       = 1.0
    brightness: float     = 0.0

    # --- panels ---
    num_panels: int  = 4
    shade_type: str  = "Normal"   # Normal | Sphere | Flat

    # --- gap between panels ---
    gap_mm: float         = 2.0
    gap_brightness: float = 0.0

    # --- wave profile ---
    waves_enabled: bool   = False
    wave_count: int       = 4
    wave_height_mm: float = 3.0

    # --- brims ---
    top_brim_height: float          = 8.0
    top_brim_thickness: float       = 5.0
    top_brim_overhang_angle: float  = 45.0
    bottom_brim_height: float       = 5.0
    bottom_brim_thickness: float    = 5.0

    # --- frames ---
    frame_width: float     = 5.0
    frame_thickness: float = 3.5

    # --- socket ---
    socket_enabled: bool       = False
    socket_inner_diam: float   = 32.5
    socket_wall: float         = 3.5
    socket_height: float       = 25.0
    socket_lip_height: float   = 3.5
    socket_lip_overhang: float = 1.5

    # --- spokes ---
    spokes_enabled: bool   = False
    spoke_count: int       = 4
    spoke_width: float     = 6.0
    spoke_thickness: float = 6.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_normals(V: np.ndarray, I: np.ndarray) -> np.ndarray:
    """Accumulate face normals per vertex and normalise."""
    N = np.zeros_like(V)
    v0, v1, v2 = V[I[:, 0]], V[I[:, 1]], V[I[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    np.add.at(N, I[:, 0], fn)
    np.add.at(N, I[:, 1], fn)
    np.add.at(N, I[:, 2], fn)
    mag = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.where(mag < 1e-12, 1.0, mag)


def _quad_indices(base: int, nrows: int, ncols: int,
                  flip: bool = False) -> np.ndarray:
    """Return (F,3) triangle indices for a (nrows x ncols) grid starting at `base`."""
    r = np.arange(nrows - 1, dtype=np.int32)
    c = np.arange(ncols - 1, dtype=np.int32)
    rr, cc = np.meshgrid(r, c, indexing='ij')  # (R-1, C-1)
    a = base + rr * ncols + cc
    b = base + rr * ncols + cc + 1
    c_ = base + (rr + 1) * ncols + cc
    d  = base + (rr + 1) * ncols + cc + 1
    if not flip:
        t1 = np.stack([a, b, c_], axis=-1).reshape(-1, 3)
        t2 = np.stack([b, d, c_], axis=-1).reshape(-1, 3)
    else:
        t1 = np.stack([a, c_, b],  axis=-1).reshape(-1, 3)
        t2 = np.stack([b, c_, d],  axis=-1).reshape(-1, 3)
    return np.concatenate([t1, t2], axis=0)


def _ring_indices(a_idx: np.ndarray, b_idx: np.ndarray,
                  flip: bool = False) -> np.ndarray:
    """Stitch two equal-length index arrays into a quad strip."""
    n = len(a_idx)
    ai  = a_idx
    ain = np.roll(a_idx, -1)
    bi  = b_idx
    bin_ = np.roll(b_idx, -1)
    if not flip:
        t1 = np.stack([ai,  bi,  ain], axis=-1)
        t2 = np.stack([ain, bi,  bin_], axis=-1)
    else:
        t1 = np.stack([ai,  ain, bi],  axis=-1)
        t2 = np.stack([ain, bin_, bi], axis=-1)
    return np.concatenate([t1, t2], axis=0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class LithophaneBuilder:
    def __init__(self, params: BuildParams):
        self.p = params

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------
    def build(self, imgs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.p.shade_type == 'Sphere':
            return self._build_sphere(imgs)
        if self.p.shade_type == 'Flat':
            return self._build_flat(imgs)
        return self._build_cylinder(imgs)

    def build_socket(self) -> Optional[Tuple]:
        p = self.p
        if not p.socket_enabled or p.socket_height <= 0:
            return None
        Vl, Il = [], []
        r_in  = p.socket_inner_diam / 2.0
        r_out = r_in + p.socket_wall
        y_bed = -float(p.bottom_brim_height) if p.bottom_brim_height > 0 else 0.0
        V, I = _build_socket_mesh(
            y_base=y_bed, y_top=float(p.socket_height),
            r_inner=r_in, r_outer=r_out,
            lip_height=p.socket_lip_height,
            lip_overhang=p.socket_lip_overhang,
        )
        Vl.append(V); Il.append(I if len(Vl) == 1 else I + len(Vl[0]))
        if p.spokes_enabled and p.spoke_count > 0:
            Vs, Is = _build_spokes_mesh(
                y_bot=y_bed, y_top=y_bed + float(p.spoke_thickness),
                r_hub=r_out, r_rim=float(p.bottom_diam) / 2.0,
                n_spokes=p.spoke_count, spoke_w=p.spoke_width,
            )
            offset = sum(len(v) for v in Vl)
            Vl.append(Vs); Il.append(Is + offset)
        V_all = np.concatenate(Vl, axis=0)
        I_all = np.concatenate(Il, axis=0)
        C_all = np.full((len(V_all), 3), 0.6, dtype=np.float64)
        N_all = _smooth_normals(V_all, I_all)
        return V_all, I_all, N_all, C_all

    # -----------------------------------------------------------------------
    # Image → thickness  (vectorised)
    # -----------------------------------------------------------------------
    def _img_to_thickness(self, img, nrows: int, ncols: int) -> np.ndarray:
        p = self.p
        data = np.asarray(
            img.resize((ncols, nrows)), dtype=np.float32) / 255.0
        data = np.clip(
            (data - 0.5) * float(p.contrast) + 0.5 + float(p.brightness),
            0.0, 1.0)
        data = np.power(data, 1.0 / max(float(p.gamma), 0.1))
        return (float(p.min_thickness)
                + (1.0 - data) * (float(p.max_thickness) - float(p.min_thickness))
                ).astype(np.float64)

    # -----------------------------------------------------------------------
    # Wave offset  (vectorised, accepts scalar or array y)
    # -----------------------------------------------------------------------
    def _wave_r(self, y, H):
        p = self.p
        if not p.waves_enabled or p.wave_count <= 0 or p.wave_height_mm <= 0:
            return np.zeros_like(y) if isinstance(y, np.ndarray) else 0.0
        phase = (y / max(H, 1e-6)) * p.wave_count * 2.0 * np.pi
        return float(p.wave_height_mm) * 0.5 * (1.0 - np.cos(phase))

    # -----------------------------------------------------------------------
    # Normal (cylinder) shade  —  fully vectorised
    # -----------------------------------------------------------------------
    def _build_cylinder(self, imgs):
        p = self.p
        H   = float(p.height)
        Rt  = float(p.top_diam)    / 2.0
        Rb  = float(p.bottom_diam) / 2.0
        t_min = float(p.min_thickness)
        t_max = float(p.max_thickness)
        NP    = p.num_panels
        ts    = 2.0 * np.pi / NP        # theta step per panel

        res   = max(float(p.resolution_mm), 0.1)
        circ  = (Rt + Rb) * 0.5 * ts
        ncols = int(np.clip(round(circ / res),  8, MAX_COLS))
        nrows = int(np.clip(round(H   / res),   8, MAX_ROWS))

        r_avg    = (Rt + Rb) * 0.5
        gap_half = (float(p.gap_mm) * 0.5) / max(r_avg, 1.0)
        gap_t    = t_min + (1.0 - float(p.gap_brightness)) * (t_max - t_min)

        # v in [0,1] top→bottom, y = H*(1-v)
        v_arr = np.linspace(0.0, 1.0, nrows, dtype=np.float64)          # (R,)
        y_arr = H * (1.0 - v_arr)                                        # (R,)
        Rv    = (1.0 - v_arr) * Rt + v_arr * Rb                         # (R,)
        wave  = self._wave_r(y_arr, H)                                   # (R,)

        u_arr = np.linspace(0.0, 1.0, ncols, dtype=np.float64)          # (C,)

        V_list, C_list, N_list, I_list = [], [], [], []
        offset = 0

        # --- track top/bottom outer rings for brim stitching ---
        top_out_rings = []
        bot_out_rings = []

        for p_idx, img in enumerate(imgs):
            arc_start = p_idx * ts + gap_half
            arc_end   = p_idx * ts + ts - gap_half
            angles = arc_start + u_arr * (arc_end - arc_start)  # (C,)

            # --- thickness grid ---
            if img is not None:
                T = self._img_to_thickness(img, nrows, ncols)   # (R, C)
            else:
                T = np.full((nrows, ncols), gap_t, dtype=np.float64)

            gray = 1.0 - np.clip(
                (T - t_min) / max(t_max - t_min, 1e-6), 0.0, 1.0)      # (R, C)

            # ---- outer surface ----
            # r_out[i,j] = Rv[i] + wave[i] + T[i,j]
            r_out = (Rv + wave)[:, None] + T                            # (R, C)

            cos_a = np.cos(angles)   # (C,)
            sin_a = np.sin(angles)   # (C,)

            x_out = r_out * cos_a                                       # (R, C)
            y_out = np.broadcast_to(y_arr[:, None], (nrows, ncols)).copy()
            z_out = r_out * sin_a

            V_out = np.stack([
                x_out.ravel(), y_out.ravel(), z_out.ravel()], axis=1)  # (R*C, 3)
            C_out = np.repeat(gray.ravel()[:, None], 3, axis=1)        # (R*C, 3)
            # normals: [cos(angle), 0, sin(angle)] per column
            nx = np.broadcast_to(cos_a[None, :], (nrows, ncols)).ravel()
            nz = np.broadcast_to(sin_a[None, :], (nrows, ncols)).ravel()
            N_out = np.stack([nx, np.zeros(nrows*ncols), nz], axis=1)

            I_out = _quad_indices(offset, nrows, ncols, flip=False)

            top_out_rings.append(np.arange(offset + 0,         offset + ncols))
            bot_out_rings.append(np.arange(offset + (nrows-1)*ncols,
                                           offset + nrows*ncols))

            V_list.append(V_out); C_list.append(C_out)
            N_list.append(N_out); I_list.append(I_out)
            offset += nrows * ncols

            # ---- inner surface ----
            r_in_arr = np.broadcast_to(Rv[:, None], (nrows, ncols)).copy()  # (R,C)
            x_in = r_in_arr * cos_a
            z_in = r_in_arr * sin_a
            V_in = np.stack([
                x_in.ravel(), y_out.ravel(), z_in.ravel()], axis=1)
            C_in = np.full((nrows * ncols, 3), 0.6, dtype=np.float64)
            N_in = np.stack([-nx, np.zeros(nrows*ncols), -nz], axis=1)
            I_in = _quad_indices(offset, nrows, ncols, flip=True)

            V_list.append(V_in); C_list.append(C_in)
            N_list.append(N_in); I_list.append(I_in)
            offset += nrows * ncols

            # ---- top edge cap (thin wall at top row) ----
            I_top = np.stack([
                np.arange(offset - nrows*ncols - nrows*ncols,
                           offset - nrows*ncols - nrows*ncols + ncols),   # outer top
                np.arange(offset - nrows*ncols,
                           offset - nrows*ncols + ncols),                  # inner top
            ], axis=0)  # we just do a ring stitch below using indices directly
            # Simpler: emit explicit cap quads
            out_top = np.arange(offset - 2*nrows*ncols,
                                offset - 2*nrows*ncols + ncols, dtype=np.int32)
            inn_top = np.arange(offset - nrows*ncols,
                                offset - nrows*ncols + ncols, dtype=np.int32)
            I_list.append(_ring_indices(out_top, inn_top, flip=True))

            # ---- bottom edge cap ----
            out_bot = np.arange(offset - 2*nrows*ncols + (nrows-1)*ncols,
                                offset - 2*nrows*ncols + nrows*ncols, dtype=np.int32)
            inn_bot = np.arange(offset - nrows*ncols + (nrows-1)*ncols,
                                offset - nrows*ncols + nrows*ncols, dtype=np.int32)
            I_list.append(_ring_indices(out_bot, inn_bot, flip=False))

        # ---- gap fillers ----
        for k in range(NP):
            t_l = k * ts - gap_half
            t_r = k * ts + gap_half
            Vg, Ig = _gap_filler_mesh(
                offset, t_l, t_r, H, Rt, Rb, gap_t, nrows,
                wave_fn=self._wave_r)
            V_list.append(Vg)
            C_list.append(np.full((len(Vg), 3), 0.6))
            N_list.append(np.tile([0.0, 0.0, 1.0], (len(Vg), 1)))  # overwritten by smooth normals
            I_list.append(Ig)
            offset += len(Vg)

        # ---- brims ----
        n_brim = max(ncols * NP, 64)
        if p.top_brim_height > 0 and p.top_brim_thickness > 0:
            Vb, Ib = _top_brim_mesh(
                offset, Rt, t_min,
                float(p.top_brim_height), float(p.top_brim_thickness),
                float(p.top_brim_overhang_angle), H, n_brim)
            V_list.append(Vb)
            C_list.append(np.full((len(Vb), 3), 0.6))
            N_list.append(np.zeros((len(Vb), 3)))  # filled by smooth normals
            I_list.append(Ib)
            offset += len(Vb)

        if p.bottom_brim_height > 0 and p.bottom_brim_thickness > 0:
            Vb, Ib = _bottom_brim_mesh(
                offset, Rb,
                float(p.bottom_brim_height), float(p.bottom_brim_thickness),
                n_brim)
            V_list.append(Vb)
            C_list.append(np.full((len(Vb), 3), 0.6))
            N_list.append(np.zeros((len(Vb), 3)))
            I_list.append(Ib)
            offset += len(Vb)

        # ---- frames ----
        if p.frame_width > 0 and p.frame_thickness > 0:
            for k in range(NP):
                theta_b = (k + 1) * ts
                Vf, If = _frame_mesh(
                    offset, theta_b,
                    H, Rt, Rb, t_min, float(p.frame_thickness),
                    float(p.frame_width), nrows)
                V_list.append(Vf)
                C_list.append(np.full((len(Vf), 3), 0.55))
                N_list.append(np.zeros((len(Vf), 3)))
                I_list.append(If)
                offset += len(Vf)

        V = np.concatenate(V_list, axis=0).astype(np.float64)
        I = np.concatenate(I_list, axis=0).astype(np.int32)
        C = np.concatenate(C_list, axis=0).astype(np.float64)
        N = _smooth_normals(V, I)
        return V, I, N, C

    # -----------------------------------------------------------------------
    # Sphere shade
    # -----------------------------------------------------------------------
    def _build_sphere(self, imgs):
        p = self.p
        R   = float(p.top_diam) / 2.0
        NP  = p.num_panels
        ts  = 2.0 * np.pi / NP
        t_min, t_max = float(p.min_thickness), float(p.max_thickness)
        res   = max(float(p.resolution_mm), 0.1)
        nrows = int(np.clip(round(np.pi * R / res), 8, MAX_ROWS))
        ncols = int(np.clip(round(R * ts / res),    8, MAX_COLS))

        phi_arr = np.linspace(0.0, np.pi, nrows, dtype=np.float64)
        u_arr   = np.linspace(0.0, 1.0,   ncols, dtype=np.float64)

        V_list, C_list, N_list, I_list = [], [], [], []
        offset = 0
        for p_idx, img in enumerate(imgs):
            if img is None:
                continue
            T    = self._img_to_thickness(img, nrows, ncols)
            gray = 1.0 - np.clip((T - t_min) / max(t_max - t_min, 1e-6), 0, 1)

            theta = p_idx * ts + u_arr * ts    # (C,)
            r_out = R + T                      # (R, C)

            sp = np.sin(phi_arr)[:, None]; cp = np.cos(phi_arr)[:, None]
            st = np.sin(theta)[None, :];  ct = np.cos(theta)[None, :]

            Vo = np.stack([
                (r_out * sp * ct).ravel(),
                (r_out * cp     ).ravel(),
                (r_out * sp * st).ravel()], axis=1)
            No = np.stack([
                (sp * ct).ravel(),
                (cp     ).ravel(),
                (sp * st).ravel()], axis=1)
            Co = np.repeat(gray.ravel()[:, None], 3, axis=1)
            Io = _quad_indices(offset, nrows, ncols)

            V_list.append(Vo); C_list.append(Co)
            N_list.append(No); I_list.append(Io)
            offset += nrows * ncols

        if not V_list:
            empty = np.zeros((0, 3), dtype=np.float64)
            return empty, np.zeros((0, 3), dtype=np.int32), empty, empty
        V = np.concatenate(V_list).astype(np.float64)
        I = np.concatenate(I_list).astype(np.int32)
        C = np.concatenate(C_list).astype(np.float64)
        N = _smooth_normals(V, I)
        return V, I, N, C

    # -----------------------------------------------------------------------
    # Flat shade
    # -----------------------------------------------------------------------
    def _build_flat(self, imgs):
        p = self.p
        W, H = float(p.bottom_diam), float(p.height)
        t_min, t_max = float(p.min_thickness), float(p.max_thickness)
        res    = max(float(p.resolution_mm), 0.1)
        nrows  = int(np.clip(round(H / res), 8, MAX_ROWS))
        ncols  = int(np.clip(round(W / res / max(p.num_panels, 1)), 8, MAX_COLS))
        pw     = W / max(p.num_panels, 1)

        xi = np.linspace(0.0, pw, ncols, dtype=np.float64)
        yi = np.linspace(0.0, H,  nrows, dtype=np.float64)
        XX, YY = np.meshgrid(xi, yi, indexing='ij')   # (C, R) → transpose
        XX = XX.T; YY = YY.T  # (R, C)

        V_list, C_list, N_list, I_list = [], [], [], []
        offset = 0
        for p_idx, img in enumerate(imgs):
            if img is None:
                continue
            T    = self._img_to_thickness(img, nrows, ncols)
            gray = 1.0 - np.clip((T - t_min) / max(t_max - t_min, 1e-6), 0, 1)
            x_off = p_idx * pw
            Vo = np.stack([
                (XX + x_off).ravel(),
                YY.ravel(),
                T.ravel()], axis=1)
            No = np.tile([0.0, 0.0, 1.0], (nrows * ncols, 1))
            Co = np.repeat(gray.ravel()[:, None], 3, axis=1)
            Io = _quad_indices(offset, nrows, ncols)
            V_list.append(Vo); C_list.append(Co)
            N_list.append(No); I_list.append(Io)
            offset += nrows * ncols

        if not V_list:
            empty = np.zeros((0, 3), dtype=np.float64)
            return empty, np.zeros((0, 3), dtype=np.int32), empty, empty
        V = np.concatenate(V_list).astype(np.float64)
        I = np.concatenate(I_list).astype(np.int32)
        C = np.concatenate(C_list).astype(np.float64)
        N = _smooth_normals(V, I)
        return V, I, N, C


# ---------------------------------------------------------------------------
# Stand-alone mesh builders (no Python loops, return V, I arrays)
# ---------------------------------------------------------------------------

def _gap_filler_mesh(offset, theta_l, theta_r, H, Rt, Rb,
                     t_gap, nrows, wave_fn) -> Tuple[np.ndarray, np.ndarray]:
    """Two-theta-column strip filling the gap. Returns (V, I)."""
    v  = np.linspace(0.0, 1.0, nrows, dtype=np.float64)
    y  = H * (1.0 - v)
    Rv = (1.0 - v) * Rt + v * Rb
    wave = wave_fn(y, H)
    r_out = Rv + wave + t_gap
    r_in  = Rv

    # 4 verts per row: left_in, left_out, right_in, right_out
    lx_in  = r_in  * np.cos(theta_l);  lz_in  = r_in  * np.sin(theta_l)
    lx_out = r_out * np.cos(theta_l);  lz_out = r_out * np.sin(theta_l)
    rx_in  = r_in  * np.cos(theta_r);  rz_in  = r_in  * np.sin(theta_r)
    rx_out = r_out * np.cos(theta_r);  rz_out = r_out * np.sin(theta_r)

    V = np.stack([
        np.stack([lx_in,  y, lz_in ], axis=1),
        np.stack([lx_out, y, lz_out], axis=1),
        np.stack([rx_in,  y, rz_in ], axis=1),
        np.stack([rx_out, y, rz_out], axis=1),
    ]).transpose(1, 0, 2).reshape(-1, 3)  # (4R, 3), order: row0:[li,lo,ri,ro], row1:...

    k   = np.arange(nrows - 1, dtype=np.int32)
    b0  = offset + k * 4
    b1  = offset + (k + 1) * 4
    # outer face
    t1  = np.stack([b0+1, b0+3, b1+1], axis=1)
    t2  = np.stack([b0+3, b1+3, b1+1], axis=1)
    # inner face
    t3  = np.stack([b0+0, b1+0, b0+2], axis=1)
    t4  = np.stack([b0+2, b1+0, b1+2], axis=1)
    # left edge
    t5  = np.stack([b0+0, b0+1, b1+0], axis=1)
    t6  = np.stack([b0+1, b1+1, b1+0], axis=1)
    # right edge
    t7  = np.stack([b0+2, b1+2, b0+3], axis=1)
    t8  = np.stack([b1+2, b1+3, b0+3], axis=1)
    # top cap (row 0)
    tb  = offset
    t9  = np.array([[tb+0, tb+1, tb+2], [tb+1, tb+3, tb+2]], dtype=np.int32)
    # bot cap
    bb  = offset + (nrows - 1) * 4
    t10 = np.array([[bb+0, bb+2, bb+1], [bb+1, bb+2, bb+3]], dtype=np.int32)

    I = np.concatenate([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10], axis=0)
    return V, I


def _annular_ring_mesh(offset, r_inner, r_outer, y,
                       n_pts, normal_dir='up') -> Tuple[np.ndarray, np.ndarray]:
    """Flat annulus at height y."""
    angs = np.linspace(0, 2 * np.pi, n_pts, endpoint=False, dtype=np.float64)
    ci, si = np.cos(angs), np.sin(angs)
    yin = np.full(n_pts, y, dtype=np.float64)

    if normal_dir == 'up':
        ny = 1.0
    else:
        ny = -1.0

    Vi = np.stack([r_inner * ci, yin, r_inner * si], axis=1)
    Vo = np.stack([r_outer * ci, yin, r_outer * si], axis=1)
    Ni = np.tile([0.0, ny, 0.0], (n_pts, 1))
    No = Ni.copy()
    V  = np.concatenate([Vi, Vo], axis=0)   # inner first, then outer

    ai  = np.arange(n_pts, dtype=np.int32)
    ain = (ai + 1) % n_pts
    bi  = ai + n_pts
    bin_ = ain + n_pts
    if ny > 0:  # up → outward winding
        t1 = np.stack([offset + ai, offset + bi, offset + ain], axis=1)
        t2 = np.stack([offset + ain, offset + bi, offset + bin_], axis=1)
    else:
        t1 = np.stack([offset + ai, offset + ain, offset + bi],  axis=1)
        t2 = np.stack([offset + ain, offset + bin_, offset + bi], axis=1)
    I = np.concatenate([t1, t2], axis=0)
    return V, I


def _vertical_cylinder_mesh(offset, r, y_bot, y_top,
                             n_pts, outward=True) -> Tuple[np.ndarray, np.ndarray]:
    angs = np.linspace(0, 2 * np.pi, n_pts, endpoint=False, dtype=np.float64)
    ci, si = np.cos(angs), np.sin(angs)
    Vbot = np.stack([r * ci, np.full(n_pts, y_bot), r * si], axis=1)
    Vtop = np.stack([r * ci, np.full(n_pts, y_top), r * si], axis=1)
    V    = np.concatenate([Vbot, Vtop], axis=0)
    ai   = np.arange(n_pts, dtype=np.int32)
    ain  = (ai + 1) % n_pts
    bi   = ai + n_pts; bin_ = ain + n_pts
    if outward:
        t1 = np.stack([offset+ai, offset+bi,   offset+ain], axis=1)
        t2 = np.stack([offset+ain, offset+bi,  offset+bin_], axis=1)
    else:
        t1 = np.stack([offset+ai, offset+ain,  offset+bi],  axis=1)
        t2 = np.stack([offset+ain, offset+bin_, offset+bi], axis=1)
    return V, np.concatenate([t1, t2])


def _top_brim_mesh(offset, Rt, t_min, brim_h, brim_t,
                   overhang_deg, y0, n_pts) -> Tuple[np.ndarray, np.ndarray]:
    """Annular top brim: outer wall + top face + underside annulus."""
    r_i  = Rt + t_min
    r_ob = r_i + brim_t
    r_ot = r_ob + brim_h * np.tan(np.radians(max(overhang_deg, 0.0)))

    Vl, Il = [], []
    off = offset

    # underside annulus at y0
    V, I = _annular_ring_mesh(off, r_i, r_ob, y0, n_pts, 'down')
    Vl.append(V); Il.append(I); off += len(V)

    # outer wall  r_ob@y0 -> r_ot@(y0+brim_h)
    V, I = _vertical_cylinder_mesh(off, r_ob, y0, y0 + brim_h, n_pts, outward=True)
    Vl.append(V); Il.append(I); off += len(V)

    # top face annulus
    V, I = _annular_ring_mesh(off, r_i, r_ot, y0 + brim_h, n_pts, 'up')
    Vl.append(V); Il.append(I); off += len(V)

    # inner wall (straight)
    V, I = _vertical_cylinder_mesh(off, r_i, y0, y0 + brim_h, n_pts, outward=False)
    Vl.append(V); Il.append(I); off += len(V)

    V_all = np.concatenate(Vl)
    I_all = np.concatenate(Il)
    return V_all, I_all


def _bottom_brim_mesh(offset, Rb, brim_h, brim_t,
                      n_pts) -> Tuple[np.ndarray, np.ndarray]:
    """Annular bottom brim."""
    r_i = Rb
    r_o = Rb + brim_t
    y_top = 0.0
    y_bot = -brim_h

    Vl, Il = [], []
    off = offset

    # top face annulus
    V, I = _annular_ring_mesh(off, r_i, r_o, y_top, n_pts, 'up')
    Vl.append(V); Il.append(I); off += len(V)

    # outer wall
    V, I = _vertical_cylinder_mesh(off, r_o, y_bot, y_top, n_pts, outward=True)
    Vl.append(V); Il.append(I); off += len(V)

    # inner wall
    V, I = _vertical_cylinder_mesh(off, r_i, y_bot, y_top, n_pts, outward=False)
    Vl.append(V); Il.append(I); off += len(V)

    # bottom face annulus
    V, I = _annular_ring_mesh(off, r_i, r_o, y_bot, n_pts, 'down')
    Vl.append(V); Il.append(I); off += len(V)

    return np.concatenate(Vl), np.concatenate(Il)


def _frame_mesh(offset, theta_b, H, Rt, Rb, t_min,
                frame_t, frame_w, nrows) -> Tuple[np.ndarray, np.ndarray]:
    """A single inter-panel pillar at angle theta_b, vectorised."""
    half_dw = (frame_w / 2.0) / max(Rt, 1e-6)
    tl = theta_b - half_dw
    tr = theta_b + half_dw

    v   = np.linspace(0.0, 1.0, nrows + 1, dtype=np.float64)
    y   = H * (1.0 - v)
    Rv  = (1.0 - v) * Rt + v * Rb
    r_base = Rv + t_min
    r_out  = r_base + frame_t

    # 4 vertices per level: left_base, right_base, left_out, right_out
    def pts(r, theta):
        return np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1)

    V = np.concatenate([
        pts(r_base, tl), pts(r_base, tr),
        pts(r_out,  tl), pts(r_out,  tr)
    ], axis=1).reshape(-1, 3)  # wrong shape
    # redo properly: interleave 4 columns
    N = nrows + 1
    V = np.empty((N * 4, 3), dtype=np.float64)
    V[0::4] = pts(r_base, tl)
    V[1::4] = pts(r_base, tr)
    V[2::4] = pts(r_out,  tl)
    V[3::4] = pts(r_out,  tr)

    k  = np.arange(nrows, dtype=np.int32)
    b0 = offset + k * 4
    b1 = offset + (k + 1) * 4

    # front face (outer)
    t1 = np.stack([b0+2, b0+3, b1+2], axis=1)
    t2 = np.stack([b0+3, b1+3, b1+2], axis=1)
    # back face (inner)
    t3 = np.stack([b0+0, b1+0, b0+1], axis=1)
    t4 = np.stack([b0+1, b1+0, b1+1], axis=1)
    # left side
    t5 = np.stack([b0+0, b0+2, b1+0], axis=1)
    t6 = np.stack([b0+2, b1+2, b1+0], axis=1)
    # right side
    t7 = np.stack([b0+1, b1+1, b0+3], axis=1)
    t8 = np.stack([b0+3, b1+1, b1+3], axis=1)

    n_v = len(V)
    # bottom cap
    tc = np.array([[offset+0, offset+1, offset+2],
                   [offset+1, offset+3, offset+2]], dtype=np.int32)
    # top cap
    et = offset + (nrows) * 4
    tt = np.array([[et+0, et+2, et+1], [et+1, et+2, et+3]], dtype=np.int32)

    I = np.concatenate([t1,t2,t3,t4,t5,t6,t7,t8,tc,tt])
    return V, I


def _build_socket_mesh(y_base, y_top, r_inner, r_outer,
                       lip_height, lip_overhang, n_seg=64
                       ) -> Tuple[np.ndarray, np.ndarray]:
    r_lip = max(r_inner - lip_overhang, 1.0)
    y_lip = y_top - lip_height

    Vl, Il = [], []
    off = 0

    # outer wall (full height)
    V, I = _vertical_cylinder_mesh(off, r_outer, y_base, y_top, n_seg, outward=True)
    Vl.append(V); Il.append(I); off += len(V)

    # inner wall (base to lip)
    V, I = _vertical_cylinder_mesh(off, r_inner, y_base, y_lip, n_seg, outward=False)
    Vl.append(V); Il.append(I); off += len(V)

    # bottom annulus
    V, I = _annular_ring_mesh(off, r_inner, r_outer, y_base, n_seg, 'down')
    Vl.append(V); Il.append(I); off += len(V)

    # lip annulus (step inward)
    V, I = _annular_ring_mesh(off, r_lip, r_inner, y_lip, n_seg, 'down')
    Vl.append(V); Il.append(I); off += len(V)

    # lip inner wall (lip to top)
    V, I = _vertical_cylinder_mesh(off, r_lip, y_lip, y_top, n_seg, outward=False)
    Vl.append(V); Il.append(I); off += len(V)

    # top annulus (r_lip to r_outer)
    V, I = _annular_ring_mesh(off, r_lip, r_outer, y_top, n_seg, 'up')
    Vl.append(V); Il.append(I); off += len(V)

    return np.concatenate(Vl), np.concatenate(Il)


def _build_spokes_mesh(y_bot, y_top, r_hub, r_rim,
                       n_spokes, spoke_w) -> Tuple[np.ndarray, np.ndarray]:
    Vl, Il = [], []
    off = 0
    ang = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False)
    hw  = spoke_w / 2.0
    for a in ang:
        tx = -np.sin(a); tz = np.cos(a)
        rx = np.cos(a);  rz = np.sin(a)
        # 8 corners of a box
        def pt(r, s, yy):
            return [r*rx + s*hw*tx, yy, r*rz + s*hw*tz]
        corners = np.array([
            pt(r_hub,-1,y_bot), pt(r_hub,+1,y_bot),
            pt(r_rim,-1,y_bot), pt(r_rim,+1,y_bot),
            pt(r_hub,-1,y_top), pt(r_hub,+1,y_top),
            pt(r_rim,-1,y_top), pt(r_rim,+1,y_top),
        ], dtype=np.float64)
        faces = np.array([
            [4,6,5],[5,6,7],
            [0,1,2],[1,3,2],
            [0,4,1],[4,5,1],
            [2,3,6],[3,7,6],
            [0,2,4],[4,2,6],
            [1,5,3],[5,7,3],
        ], dtype=np.int32) + off
        Vl.append(corners); Il.append(faces); off += 8
    return np.concatenate(Vl), np.concatenate(Il)
