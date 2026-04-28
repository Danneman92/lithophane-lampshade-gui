"""LithophaneMaker-parity lamp builder.

Key design decisions matching lithophanemaker.com:
  - Resolution driven by mm/px, not fixed row/col counts.
  - Gamma-corrected brightness → thickness mapping.
  - Per-panel contrast + brightness adjustments.
  - Smooth per-vertex normals computed from face geometry.
  - Configurable gap between panels (width + brightness).
  - Optional wave profile on lamp body.
  - Brim is a clean annular band (no inner disc) — lamp opening stays open.
  - Socket / interface ring + spokes built as *separate* mesh objects so
    they can be exported as separate STL files without topology conflicts.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from geometry_utils import (
    stitch_rings, stitch_wall,
    build_ring_xyz_at_radius, compute_smooth_normals,
)


@dataclass
class BuildParams:
    # --- shape ---
    height: float       = 163.0
    top_diam: float     = 170.0
    bottom_diam: float  = 200.0

    # --- lithophane image quality ---
    min_thickness: float  = 0.8
    max_thickness: float  = 3.0
    resolution_mm: float  = 0.5   # mm per pixel (smaller = finer, larger STL)
    gamma: float          = 2.2   # perceptual brightness correction
    contrast: float       = 1.0   # multiplier on pixel contrast [0.5 – 2.0]
    brightness: float     = 0.0   # additive offset [-0.5 – +0.5]

    # --- panels ---
    num_panels: int   = 4
    shade_type: str   = "Normal"  # Normal | Sphere | Flat

    # --- gap between panels ---
    gap_mm: float         = 2.0   # angular width of gap at shade surface
    gap_brightness: float = 0.0   # 0 = max thickness (dark), 1 = min thickness (clear)

    # --- wave profile ---
    waves_enabled: bool  = False
    wave_count: int      = 4
    wave_height_mm: float = 3.0

    # --- top brim (lip / collar) ---
    top_brim_height: float     = 8.0
    top_brim_thickness: float  = 5.0
    top_brim_overhang_angle: float = 45.0  # degrees; 0 = straight wall

    # --- bottom brim (base ring) ---
    bottom_brim_height: float     = 5.0
    bottom_brim_thickness: float  = 5.0

    # --- frames / pillars between panels ---
    frame_width: float     = 5.0
    frame_thickness: float = 3.5

    # --- socket adapter (interface to lamp) ---
    socket_enabled: bool         = False
    socket_inner_diam: float     = 32.5  # E27 ≈ 26 mm, E14 ≈ 17 mm
    socket_wall: float           = 3.5
    socket_height: float         = 25.0
    socket_lip_height: float     = 3.5
    socket_lip_overhang: float   = 1.5

    # --- spokes ---
    spokes_enabled: bool  = False
    spoke_count: int      = 4
    spoke_width: float    = 6.0
    spoke_thickness: float = 6.0


class LithophaneBuilder:
    def __init__(self, params: BuildParams):
        self.p = params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, imgs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (V, I, N, C) for the shade + brim + frames."""
        if self.p.shade_type == "Sphere":
            return self._build_sphere(imgs)
        if self.p.shade_type == "Flat":
            return self._build_flat(imgs)
        return self._build_cylinder(imgs)

    def build_socket(self, imgs=None) -> Optional[Tuple]:
        """Return (V, I, N, C) for socket+spokes only, or None if disabled."""
        p = self.p
        if not p.socket_enabled or p.socket_height <= 0:
            return None
        vertices, colors, normals, indices = [], [], [], []
        mid_gray = (0.6, 0.6, 0.6)
        Rb = float(p.bottom_diam) / 2.0
        y_bed = -float(p.bottom_brim_height) if p.bottom_brim_height > 0 else 0.0
        r_sock_in  = p.socket_inner_diam / 2.0
        r_sock_out = r_sock_in + p.socket_wall
        self._build_socket(
            vertices, colors, normals, indices,
            y_base=y_bed, y_top=float(p.socket_height),
            r_inner=r_sock_in, r_outer=r_sock_out,
            lip_height=p.socket_lip_height, lip_overhang=p.socket_lip_overhang,
            color=mid_gray,
        )
        if p.spokes_enabled and p.spoke_count > 0:
            self._build_spokes(
                vertices, colors, normals, indices,
                y_bot=y_bed, y_top=y_bed + float(p.spoke_thickness),
                r_hub=r_sock_out, r_rim=Rb,
                n_spokes=p.spoke_count, spoke_w=p.spoke_width, color=mid_gray,
            )
        V = np.array(vertices, dtype=np.float64)
        I = np.array(indices,  dtype=np.int32)
        C = np.array(colors,   dtype=np.float64)
        N = compute_smooth_normals(V, I) if len(I) > 0 else np.zeros_like(V)
        return V, I, N, C

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _img_to_thickness(self, img, nrows: int, ncols: int) -> np.ndarray:
        """Convert a PIL image to a (nrows, ncols) thickness array.

        Pipeline (matches LithophaneMaker):
          1. Resize to target resolution
          2. Apply contrast + brightness
          3. Gamma-decode (perceptual → linear light)
          4. Map: bright pixel → thin (transparent), dark → thick (opaque)
        """
        p = self.p
        data = np.asarray(img.resize((ncols, nrows)), dtype=np.float32) / 255.0
        # contrast & brightness
        data = np.clip((data - 0.5) * float(p.contrast) + 0.5 + float(p.brightness), 0.0, 1.0)
        # gamma correction
        gamma = max(float(p.gamma), 0.1)
        data = np.power(data, 1.0 / gamma)
        # bright → thin, dark → thick
        T = float(p.min_thickness) + (1.0 - data) * (float(p.max_thickness) - float(p.min_thickness))
        return T.astype(np.float64)

    def _wave_offset(self, y: float, H: float) -> float:
        """Radial offset due to wave profile at height y."""
        p = self.p
        if not p.waves_enabled or p.wave_count <= 0 or p.wave_height_mm <= 0:
            return 0.0
        phase = (y / H) * p.wave_count * 2.0 * np.pi
        return float(p.wave_height_mm) * 0.5 * (1.0 - np.cos(phase))

    # ------------------------------------------------------------------
    # Cylinder (Normal) shade
    # ------------------------------------------------------------------
    def _build_cylinder(self, imgs):
        p = self.p
        H   = float(p.height)
        Rt  = float(p.top_diam)    / 2.0
        Rb  = float(p.bottom_diam) / 2.0
        t_min = float(p.min_thickness)
        t_max = float(p.max_thickness)
        num_panels  = p.num_panels
        theta_step  = 2.0 * np.pi / num_panels

        # Resolution: compute nrows/ncols from mm/px
        circ_per_panel = (Rt + Rb) * 0.5 * theta_step  # arc length per panel
        res  = max(float(p.resolution_mm), 0.1)
        ncols = max(int(round(circ_per_panel / res)), 8)
        nrows = max(int(round(H / res)), 8)

        # Gap angular half-width in radians at average radius
        r_avg    = (Rt + Rb) * 0.5
        gap_half = (float(p.gap_mm) * 0.5) / max(r_avg, 1.0)
        gap_t    = t_min + (1.0 - float(p.gap_brightness)) * (t_max - t_min)

        panels_present = [img is not None for img in imgs]

        vertices, colors, normals, indices = [], [], [], []
        top_outer_ring, top_inner_ring = [], []
        bot_outer_ring, bot_inner_ring = [], []

        panel_rows_angles: List[List[np.ndarray]] = []
        panel_rows_radii:  List[List[np.ndarray]] = []
        panel_rows_y:      List[np.ndarray]        = []

        for p_idx, img in enumerate(imgs):
            # --- precompute thickness grid ---
            if img is not None:
                T = self._img_to_thickness(img, nrows, ncols)  # (nrows, ncols)
                gray = 1.0 - np.clip(
                    (T - t_min) / max(t_max - t_min, 1e-6), 0.0, 1.0)
            else:
                T    = np.full((nrows, ncols), gap_t, dtype=np.float64)
                gray = np.full((nrows, ncols), 0.6,   dtype=np.float64)

            row_angles_list, row_radii_list = [], []
            row_y = np.empty(nrows, dtype=np.float64)

            base = len(vertices)
            for i in range(nrows):
                v   = i / max(nrows - 1, 1)
                y   = H * (1.0 - v)
                row_y[i] = y
                Rv  = (1.0 - v) * Rt + v * Rb
                wave_r = self._wave_offset(y, H)

                # Panel arc: leave gap_half on each side
                u_arr   = np.linspace(0.0, 1.0, ncols, dtype=np.float64)
                # map u into [gap_half, theta_step - gap_half]
                arc_start = p_idx * theta_step + gap_half
                arc_end   = p_idx * theta_step + theta_step - gap_half
                angles    = arc_start + u_arr * (arc_end - arc_start)
                r_out_row = Rv + wave_r + T[i, :]

                for j in range(ncols):
                    cx = r_out_row[j] * np.cos(angles[j])
                    cz = r_out_row[j] * np.sin(angles[j])
                    vertices.append([cx, y, cz])
                    colors.append([float(gray[i, j])] * 3)
                    # radially outward normal — correct for curved surface
                    normals.append([np.cos(angles[j]), 0.0, np.sin(angles[j])])

                row_angles_list.append(angles)
                row_radii_list.append(r_out_row)

            panel_rows_angles.append(row_angles_list)
            panel_rows_radii.append(row_radii_list)
            panel_rows_y.append(row_y)

            # outer surface quads
            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base + i * ncols + j
                    b = base + i * ncols + (j + 1)
                    c = base + (i + 1) * ncols + j
                    d = base + (i + 1) * ncols + (j + 1)
                    indices.append([a, b, c])
                    indices.append([b, d, c])

            top_outer_ring.extend([base + j for j in range(ncols - 1, -1, -1)])
            bot_outer_ring.extend([base + (nrows - 1) * ncols + j for j in range(ncols - 1, -1, -1)])

            # inner surface (base cylinder at Rv, no image displacement)
            base_in = len(vertices)
            for i in range(nrows):
                v   = i / max(nrows - 1, 1)
                y   = H * (1.0 - v)
                Rv  = (1.0 - v) * Rt + v * Rb
                u_arr  = np.linspace(0.0, 1.0, ncols, dtype=np.float64)
                arc_start = p_idx * theta_step + gap_half
                arc_end   = p_idx * theta_step + theta_step - gap_half
                angles = arc_start + u_arr * (arc_end - arc_start)
                cx = Rv * np.cos(angles)
                cz = Rv * np.sin(angles)
                for j in range(ncols):
                    vertices.append([cx[j], y, cz[j]])
                    colors.append([0.6, 0.6, 0.6])
                    # radially inward normal
                    normals.append([-np.cos(angles[j]), 0.0, -np.sin(angles[j])])

            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base_in + i * ncols + j
                    b = base_in + i * ncols + (j + 1)
                    c = base_in + (i + 1) * ncols + j
                    d = base_in + (i + 1) * ncols + (j + 1)
                    indices.append([a, c, b])
                    indices.append([b, c, d])

            top_inner_ring.extend([base_in + j for j in range(ncols - 1, -1, -1)])
            bot_inner_ring.extend([base_in + (nrows - 1) * ncols + j for j in range(ncols - 1, -1, -1)])

        # Cap the thin panel wall at top and bottom
        if top_outer_ring and top_inner_ring:
            stitch_rings(indices, top_inner_ring, top_outer_ring, outward=True)
        if bot_outer_ring and bot_inner_ring:
            stitch_rings(indices, bot_outer_ring, bot_inner_ring, outward=False)

        mid_gray = (0.6, 0.6, 0.6)

        # ------------------------------------------------------------------
        # Gap fillers between panels
        # Each gap is a thin strip at theta = k*theta_step ± gap_half
        # ------------------------------------------------------------------
        for k in range(num_panels):
            theta_center = k * theta_step  # left edge of this panel
            # left gap (between panel k-1 and panel k)
            t_left  = theta_center - gap_half
            t_right = theta_center + gap_half
            self._build_gap_filler(
                vertices, colors, normals, indices,
                theta_left=t_left, theta_right=t_right,
                H=H, Rt=Rt, Rb=Rb, t_gap=gap_t, nrows=nrows,
                wave_fn=self._wave_offset, color=mid_gray,
            )

        # ------------------------------------------------------------------
        # Top brim  (annular band, NO inner disc — lamp opening stays open)
        # ------------------------------------------------------------------
        if p.top_brim_height > 0 and p.top_brim_thickness > 0 and any(panels_present):
            self._build_top_brim(
                vertices, colors, normals, indices,
                Rt=Rt, t_min=t_min,
                brim_h=float(p.top_brim_height),
                brim_t=float(p.top_brim_thickness),
                overhang_deg=float(p.top_brim_overhang_angle),
                y0=H,
                n_pts=max(ncols * num_panels, 64),
                top_outer_ring=top_outer_ring,
                color=mid_gray,
            )

        # ------------------------------------------------------------------
        # Bottom brim (annular band, NO inner disc)
        # ------------------------------------------------------------------
        if p.bottom_brim_height > 0 and p.bottom_brim_thickness > 0 and any(panels_present):
            self._build_bottom_brim(
                vertices, colors, normals, indices,
                Rb=Rb, t_min=t_min,
                brim_h=float(p.bottom_brim_height),
                brim_t=float(p.bottom_brim_thickness),
                n_pts=max(ncols * num_panels, 64),
                bot_outer_ring=bot_outer_ring,
                color=mid_gray,
            )

        # ------------------------------------------------------------------
        # Frame pillars between panels
        # ------------------------------------------------------------------
        if p.frame_width > 0 and p.frame_thickness > 0:
            self._build_frames(
                vertices, colors, normals, indices,
                H=H, Rt=Rt, Rb=Rb, t_min=t_min,
                nrows=nrows,
                panel_rows_angles=panel_rows_angles,
                panel_rows_radii=panel_rows_radii,
                panel_rows_y=panel_rows_y,
                theta_step=theta_step,
            )

        # Socket is built separately via build_socket()
        V = np.array(vertices, dtype=np.float64)
        I = np.array(indices,  dtype=np.int32)
        C = np.array(colors,   dtype=np.float64)
        N = compute_smooth_normals(V, I) if len(I) > 0 else np.zeros_like(V)
        return V, I, N, C

    # ------------------------------------------------------------------
    def _build_gap_filler(self, vertices, colors, normals, indices,
                          theta_left, theta_right, H, Rt, Rb, t_gap, nrows,
                          wave_fn, color):
        """Fill the angular gap between two panels with a flat strip."""
        base = len(vertices)
        for i in range(nrows):
            v  = i / max(nrows - 1, 1)
            y  = H * (1.0 - v)
            Rv = (1.0 - v) * Rt + v * Rb
            wave_r = wave_fn(y, H)
            r_out = Rv + wave_r + t_gap
            r_in  = Rv
            for theta in [theta_left, theta_right]:
                for r in [r_in, r_out]:
                    cx = r * np.cos(theta)
                    cz = r * np.sin(theta)
                    vertices.append([cx, y, cz])
                    colors.append(list(color))
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])
        # 4 vertices per row: [left_in, left_out, right_in, right_out]
        for i in range(nrows - 1):
            b0 = base + i * 4
            b1 = base + (i + 1) * 4
            # outer face (left_out, right_out)
            indices.append([b0+1, b0+3, b1+1]); indices.append([b0+3, b1+3, b1+1])
            # inner face
            indices.append([b0+0, b1+0, b0+2]); indices.append([b0+2, b1+0, b1+2])
            # left edge
            indices.append([b0+0, b0+1, b1+0]); indices.append([b0+1, b1+1, b1+0])
            # right edge
            indices.append([b0+2, b1+2, b0+3]); indices.append([b1+2, b1+3, b0+3])
        # top cap
        tb = base
        indices.append([tb+0, tb+1, tb+2]); indices.append([tb+1, tb+3, tb+2])
        # bottom cap
        bb = base + (nrows - 1) * 4
        indices.append([bb+0, bb+2, bb+1]); indices.append([bb+1, bb+2, bb+3])

    # ------------------------------------------------------------------
    def _build_top_brim(self, vertices, colors, normals, indices,
                        Rt, t_min, brim_h, brim_t, overhang_deg, y0,
                        n_pts, top_outer_ring, color):
        """Annular top brim.  r_inner = Rt+t_min, r_outer flares outward.
        overhang_deg: if >0, the outer wall leans outward (like LithophaneMaker
        'overhang angle') so no support is needed."""
        r_inner = Rt + t_min
        r_outer_bot = r_inner + brim_t
        # outer radius at top after overhang
        overhang_extra = brim_h * np.tan(np.radians(max(overhang_deg, 0.0)))
        r_outer_top = r_outer_bot + overhang_extra

        def ring(r, y, ntype='up'):
            s = len(vertices)
            angs = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            for ang in angs:
                vertices.append([r * np.cos(ang), y, r * np.sin(ang)])
                colors.append(list(color))
                if ntype == 'outward':
                    normals.append([np.cos(ang), 0.0, np.sin(ang)])
                elif ntype == 'down':
                    normals.append([0.0, -1.0, 0.0])
                else:
                    normals.append([0.0, 1.0, 0.0])
            return list(range(s, s + n_pts))

        def stitch(a, b, outward=True):
            n = len(a)
            for ii in range(n):
                jj = (ii + 1) % n
                if outward:
                    indices.append([a[ii], b[ii], a[jj]])
                    indices.append([a[jj], b[ii], b[jj]])
                else:
                    indices.append([a[ii], a[jj], b[ii]])
                    indices.append([a[jj], b[jj], b[ii]])

        # Underside ring at r_inner, y0 — stitch to panel tops
        base_inner = ring(r_inner, y0, 'down')
        base_outer = ring(r_outer_bot, y0, 'down')

        # Stitch panel top edges to base_inner slice
        if top_outer_ring:
            m = len(base_inner)
            # panel outer ring may have fewer vertices — map by angle
            # simplest: stitch directly if counts match, else skip
            if len(top_outer_ring) == m:
                stitch_rings(indices, top_outer_ring, base_inner, outward=False)

        # Flat underside annulus
        stitch(base_inner, base_outer, outward=False)

        # Outer wall (with overhang)
        top_outer = ring(r_outer_top, y0 + brim_h, 'outward')
        stitch(base_outer, top_outer, outward=False)

        # Top face annulus
        top_inner = ring(r_inner, y0 + brim_h, 'up')
        stitch(top_inner, top_outer, outward=False)

    # ------------------------------------------------------------------
    def _build_bottom_brim(self, vertices, colors, normals, indices,
                           Rb, t_min, brim_h, brim_t, n_pts,
                           bot_outer_ring, color):
        """Annular bottom brim.  No inner disc — floor opening stays open."""
        r_inner = Rb
        r_outer = Rb + brim_t
        y_top   = 0.0
        y_bot   = -brim_h

        def ring(r, y, ntype='up'):
            s = len(vertices)
            angs = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            for ang in angs:
                vertices.append([r * np.cos(ang), y, r * np.sin(ang)])
                colors.append(list(color))
                if ntype == 'outward':
                    normals.append([np.cos(ang), 0.0, np.sin(ang)])
                elif ntype == 'down':
                    normals.append([0.0, -1.0, 0.0])
                elif ntype == 'inward':
                    normals.append([-np.cos(ang), 0.0, -np.sin(ang)])
                else:
                    normals.append([0.0, 1.0, 0.0])
            return list(range(s, s + n_pts))

        def stitch(a, b, outward=True):
            n = len(a)
            for ii in range(n):
                jj = (ii + 1) % n
                if outward:
                    indices.append([a[ii], b[ii], a[jj]])
                    indices.append([a[jj], b[ii], b[jj]])
                else:
                    indices.append([a[ii], a[jj], b[ii]])
                    indices.append([a[jj], b[jj], b[ii]])

        inner_top = ring(r_inner, y_top, 'inward')
        outer_top = ring(r_outer, y_top, 'outward')

        # Stitch panel bottom edges to inner_top
        if bot_outer_ring and len(bot_outer_ring) == len(inner_top):
            stitch_rings(indices, bot_outer_ring, inner_top, outward=False)

        # Top face annulus (faces down into shade)
        stitch(inner_top, outer_top, outward=False)

        # Inner vertical wall
        inner_bot = ring(r_inner, y_bot, 'inward')
        stitch(inner_bot, inner_top, outward=True)

        # Outer vertical wall
        outer_bot = ring(r_outer, y_bot, 'outward')
        stitch(outer_top, outer_bot, outward=False)

        # Bottom face annulus
        stitch(inner_bot, outer_bot, outward=True)

    # ------------------------------------------------------------------
    def _build_frames(self, vertices, colors, normals, indices,
                      H, Rt, Rb, t_min, nrows,
                      panel_rows_angles, panel_rows_radii, panel_rows_y,
                      theta_step):
        p = self.p
        num_panels = p.num_panels
        r_avg = (Rt + Rb) * 0.5
        gap_half = (float(p.gap_mm) * 0.5) / max(r_avg, 1.0)
        ys = np.linspace(0.0, H, nrows + 1, dtype=np.float64)

        for k in range(num_panels):
            theta_boundary = (k + 1) * theta_step
            half_dw   = (p.frame_width / 2.0) / max(Rt, 1e-6)
            theta_left  = theta_boundary - half_dw
            theta_right = theta_boundary + half_dw

            left_idx  = k % num_panels
            right_idx = (k + 1) % num_panels
            left_y    = panel_rows_y[left_idx]
            right_y   = panel_rows_y[right_idx]

            pillar_base = len(vertices)
            pillar_r_by_level = []

            for y_val in ys:
                v  = 0.0 if H <= 0 else (H - y_val) / H
                Rv = (1.0 - v) * Rt + v * Rb
                baseline = Rv + t_min

                # interpolate outer radius from adjacent panels
                def _interp_r(rows_angles, rows_radii, py, theta):
                    if py.size == 0:
                        return baseline
                    li = int(np.argmin(np.abs(py - y_val)))
                    la = np.asarray(rows_angles[li]).ravel()
                    lr = np.asarray(rows_radii[li]).ravel()
                    if la.size < 2:
                        return baseline
                    idx = np.argsort(la)
                    la, lr = la[idx], lr[idx]
                    return float(np.interp(np.clip(theta, la.min(), la.max()), la, lr))

                r_left  = _interp_r(panel_rows_angles[left_idx],  panel_rows_radii[left_idx],  left_y,  theta_left)
                r_right = _interp_r(panel_rows_angles[right_idx], panel_rows_radii[right_idx], right_y, theta_right)
                r_out   = max(baseline + p.frame_thickness, r_left + 1e-3, r_right + 1e-3)
                pillar_r_by_level.append((r_left, r_right, r_out))

                for idx2, (r, th) in enumerate([
                        (r_left,  theta_left),
                        (r_right, theta_right),
                        (r_out,   theta_left),
                        (r_out,   theta_right)]):
                    vertices.append([r * np.cos(th), y_val, r * np.sin(th)])
                    colors.append([0.55, 0.55, 0.55])
                    normals.append([-np.cos(th), 0, -np.sin(th)] if idx2 < 2
                                   else [np.cos(th), 0, np.sin(th)])

            n_steps = len(ys)
            for s in range(n_steps - 1):
                b0 = pillar_base + s * 4
                b1 = pillar_base + (s + 1) * 4
                indices.append([b0+0, b1+0, b0+1]); indices.append([b1+0, b1+1, b0+1])
                indices.append([b0+2, b0+3, b1+2]); indices.append([b1+2, b0+3, b1+3])
                indices.append([b0+0, b0+2, b1+0]); indices.append([b1+0, b0+2, b1+2])
                indices.append([b0+1, b1+1, b0+3]); indices.append([b1+1, b1+3, b0+3])

            # bottom cap
            cb = len(vertices)
            for ci in range(4):
                vertices.append(list(vertices[pillar_base + ci]))
                colors.append([0.55, 0.55, 0.55])
                normals.append([0.0, -1.0, 0.0])
            indices.append([cb+0, cb+1, cb+2]); indices.append([cb+1, cb+3, cb+2])

            # top cap
            et = pillar_base + (n_steps - 1) * 4
            ct = len(vertices)
            for ci in range(4):
                vertices.append(list(vertices[et + ci]))
                colors.append([0.55, 0.55, 0.55])
                normals.append([0.0, 1.0, 0.0])
            indices.append([ct+0, ct+2, ct+1]); indices.append([ct+1, ct+2, ct+3])

    # ------------------------------------------------------------------
    # Socket & spokes
    # ------------------------------------------------------------------
    def _build_socket(self, vertices, colors, normals, indices,
                      y_base, y_top, r_inner, r_outer,
                      lip_height, lip_overhang, color, n_seg=64):
        angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        r_lip  = max(r_inner - lip_overhang, 1.0)
        y_lip  = y_top - lip_height

        def add_ring(r, y, nrm_fn):
            s = len(vertices)
            for a in angles:
                vertices.append([r * np.cos(a), y, r * np.sin(a)])
                colors.append(list(color))
                normals.append(list(nrm_fn(a)))
            return list(range(s, s + n_seg))

        up      = lambda a: [0.0,  1.0, 0.0]
        dn      = lambda a: [0.0, -1.0, 0.0]
        r_out_n = lambda a: [ np.cos(a), 0.0,  np.sin(a)]
        r_in_n  = lambda a: [-np.cos(a), 0.0, -np.sin(a)]

        ri_base = add_ring(r_inner, y_base, r_in_n)
        ro_base = add_ring(r_outer, y_base, r_out_n)
        ri_lip  = add_ring(r_inner, y_lip,  r_in_n)
        ro_lip  = add_ring(r_outer, y_lip,  r_out_n)
        rl_lip  = add_ring(r_lip,   y_lip,  dn)
        rl_top  = add_ring(r_lip,   y_top,  r_in_n)
        ri_top  = add_ring(r_inner, y_top,  up)
        ro_top  = add_ring(r_outer, y_top,  up)

        def quad(a, b, c, d):
            indices.append([a, b, c]); indices.append([a, c, d])

        for i in range(n_seg):
            j = (i + 1) % n_seg
            quad(ro_base[i], ro_top[i],  ro_top[j],  ro_base[j])
            quad(ri_lip[j],  ri_base[j], ri_base[i], ri_lip[i])
            quad(ro_base[i], ri_base[i], ri_base[j], ro_base[j])
            quad(rl_lip[j],  rl_top[j],  rl_top[i],  rl_lip[i])
            quad(ri_lip[i],  rl_lip[i],  rl_lip[j],  ri_lip[j])
            quad(rl_top[i],  ri_top[i],  ri_top[j],  rl_top[j])
            quad(ri_top[i],  ro_top[i],  ro_top[j],  ri_top[j])

    def _build_spokes(self, vertices, colors, normals, indices,
                      y_bot, y_top, r_hub, r_rim, n_spokes, spoke_w, color):
        angle_step = 2.0 * np.pi / n_spokes
        half_w = spoke_w / 2.0
        for k in range(n_spokes):
            a_ctr = k * angle_step
            tx = -np.sin(a_ctr); tz = np.cos(a_ctr)
            rx =  np.cos(a_ctr); rz = np.sin(a_ctr)

            def pt(r, side, yy):
                return [r * rx + side * half_w * tx, yy, r * rz + side * half_w * tz]

            corners = [
                pt(r_hub, -1, y_bot), pt(r_hub, +1, y_bot),
                pt(r_rim, -1, y_bot), pt(r_rim, +1, y_bot),
                pt(r_hub, -1, y_top), pt(r_hub, +1, y_top),
                pt(r_rim, -1, y_top), pt(r_rim, +1, y_top),
            ]
            base = len(vertices)
            for cx, cy, cz in corners:
                vertices.append([cx, cy, cz])
                colors.append(list(color))
                normals.append([0.0, 1.0, 0.0])

            def f(a, b, c): indices.append([base + a, base + b, base + c])
            f(4, 6, 5); f(5, 6, 7)
            f(0, 1, 2); f(1, 3, 2)
            f(0, 4, 1); f(4, 5, 1)
            f(2, 3, 6); f(3, 7, 6)
            f(0, 2, 4); f(4, 2, 6)
            f(1, 5, 3); f(5, 7, 3)

    # ------------------------------------------------------------------
    # Sphere mode
    # ------------------------------------------------------------------
    def _build_sphere(self, imgs):
        p = self.p
        R  = float(p.top_diam) / 2.0
        num_panels  = p.num_panels
        theta_step  = 2.0 * np.pi / num_panels
        t_min, t_max = float(p.min_thickness), float(p.max_thickness)
        res  = max(float(p.resolution_mm), 0.1)
        circ = np.pi * R
        nrows = max(int(round(circ / res)), 8)
        ncols = max(int(round(R * theta_step / res)), 8)
        vertices, colors, normals, indices = [], [], [], []
        for p_idx, img in enumerate(imgs):
            if img is None:
                continue
            T    = self._img_to_thickness(img, nrows, ncols)
            gray = 1.0 - np.clip((T - t_min) / max(t_max - t_min, 1e-6), 0, 1)
            base = len(vertices)
            for i in range(nrows):
                phi = np.pi * i / max(nrows - 1, 1)
                for j in range(ncols):
                    u     = j / max(ncols - 1, 1)
                    theta = p_idx * theta_step + u * theta_step
                    r = R + T[i, j]
                    vertices.append([r*np.sin(phi)*np.cos(theta),
                                     r*np.cos(phi),
                                     r*np.sin(phi)*np.sin(theta)])
                    colors.append([float(gray[i, j])] * 3)
                    normals.append([np.sin(phi)*np.cos(theta),
                                    np.cos(phi),
                                    np.sin(phi)*np.sin(theta)])
            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base + i*ncols + j
                    b = base + i*ncols + (j+1)
                    c = base + (i+1)*ncols + j
                    d = base + (i+1)*ncols + (j+1)
                    indices.append([a, b, c]); indices.append([b, d, c])
        V = np.array(vertices, dtype=np.float64)
        I = np.array(indices,  dtype=np.int32)
        C = np.array(colors,   dtype=np.float64)
        N = compute_smooth_normals(V, I) if len(I) > 0 else np.zeros_like(V)
        return V, I, N, C

    # ------------------------------------------------------------------
    # Flat mode
    # ------------------------------------------------------------------
    def _build_flat(self, imgs):
        p = self.p
        W, H = float(p.bottom_diam), float(p.height)
        t_min, t_max = float(p.min_thickness), float(p.max_thickness)
        res      = max(float(p.resolution_mm), 0.1)
        nrows    = max(int(round(H / res)), 8)
        ncols    = max(int(round(W / res / max(p.num_panels, 1))), 8)
        panel_w  = W / max(p.num_panels, 1)
        vertices, colors, normals, indices = [], [], [], []
        for p_idx, img in enumerate(imgs):
            if img is None:
                continue
            T    = self._img_to_thickness(img, nrows, ncols)
            gray = 1.0 - np.clip((T - t_min) / max(t_max - t_min, 1e-6), 0, 1)
            x_off = p_idx * panel_w
            base  = len(vertices)
            for i in range(nrows):
                for j in range(ncols):
                    x = x_off + j / max(ncols - 1, 1) * panel_w
                    y = i / max(nrows - 1, 1) * H
                    z = T[i, j]
                    vertices.append([x, y, z])
                    colors.append([float(gray[i, j])] * 3)
                    normals.append([0, 0, 1])
            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base + i*ncols + j
                    b = base + i*ncols + (j+1)
                    c = base + (i+1)*ncols + j
                    d = base + (i+1)*ncols + (j+1)
                    indices.append([a, b, c]); indices.append([b, d, c])
        V = np.array(vertices, dtype=np.float64)
        I = np.array(indices,  dtype=np.int32)
        C = np.array(colors,   dtype=np.float64)
        N = compute_smooth_normals(V, I) if len(I) > 0 else np.zeros_like(V)
        return V, I, N, C
