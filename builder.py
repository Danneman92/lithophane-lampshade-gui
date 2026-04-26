from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from geometry_utils import stitch_rings, stitch_wall, build_ring_xyz_at_radius


@dataclass
class BuildParams:
    height: float
    top_diam: float
    bottom_diam: float
    min_thickness: float
    max_thickness: float
    top_brim_height: float
    top_brim_thickness: float
    bottom_brim_height: float
    bottom_brim_thickness: float
    frame_width: float
    frame_thickness: float
    nrows: int = 120
    ncols: int = 160
    num_panels: int = 4
    shade_type: str = "Normal"
    # Sprocket
    sprocket_enabled: bool = False
    sprocket_teeth: int = 24
    sprocket_tooth_height: float = 3.0   # mm above top-brim top face
    sprocket_tooth_width: float = 0.5    # fraction of inter-tooth arc
    # Lamp socket
    socket_enabled: bool = False
    socket_inner_diam: float = 26.0      # E27 bulb neck ~26 mm
    socket_wall: float = 2.5
    socket_height: float = 20.0


class LithophaneBuilder:
    def __init__(self, params: BuildParams):
        self.p = params

    def build(self, imgs: List[Optional["Image.Image"]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.p.shade_type == "Sphere":
            return self._build_sphere(imgs)
        if self.p.shade_type == "Flat":
            return self._build_flat(imgs)
        return self._build_cylinder(imgs)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_cap_ring(vertices, colors, normals,
                      xzs, y, color, normal_y):
        """Add a horizontal ring of vertices and return their indices."""
        start = len(vertices)
        for x, z in xzs:
            vertices.append([x, y, z])
            colors.append(list(color))
            normals.append([0.0, float(normal_y), 0.0])
        return list(range(start, start + len(xzs)))

    @staticmethod
    def _ring_xz(r, n_pts, offset_angle=0.0):
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset_angle
        return [(r * np.cos(a), r * np.sin(a)) for a in angles]

    # ------------------------------------------------------------------
    # cylinder builder
    # ------------------------------------------------------------------
    def _build_cylinder(self, imgs):
        p = self.p
        H = float(p.height)
        Rt = float(p.top_diam) / 2.0
        Rb = float(p.bottom_diam) / 2.0
        nrows, ncols, num_panels = p.nrows, p.ncols, p.num_panels
        theta_step = 2.0 * np.pi / num_panels
        t_min = float(p.min_thickness)
        t_max = float(p.max_thickness)

        vertices, colors, normals, indices = [], [], [], []
        top_outer_ring, top_inner_ring, bot_outer_ring, bot_inner_ring = [], [], [], []
        panels_present = [img is not None for img in imgs]

        panel_rows_angles: List[List[np.ndarray]] = []
        panel_rows_radii: List[List[np.ndarray]]  = []
        panel_rows_y: List[np.ndarray]             = []

        def panel_base_index():
            return len(vertices)

        # ---- outer + inner shells ----
        for p_idx, img in enumerate(imgs):
            if img is None:
                panel_rows_angles.append([np.array([], dtype=np.float64) for _ in range(nrows)])
                panel_rows_radii.append( [np.array([], dtype=np.float64) for _ in range(nrows)])
                panel_rows_y.append(np.linspace(H, 0.0, nrows, dtype=np.float64))
                continue

            data = np.asarray(img.resize((ncols, nrows)), dtype=np.float32) / 255.0
            T    = t_min + (1.0 - data) * (t_max - t_min)
            gray = 1.0 - ((T - t_min) / max(t_max - t_min, 1e-6))

            row_angles_list, row_radii_list = [], []
            row_y = np.empty(nrows, dtype=np.float64)

            base = panel_base_index()
            for i in range(nrows):
                v   = i / (nrows - 1)
                y   = H - v * H
                row_y[i] = y
                Rv  = (1.0 - v) * Rt + v * Rb
                u_prime = 1.0 - (np.arange(ncols, dtype=np.float64) / float(ncols - 1))
                angles  = p_idx * theta_step + u_prime * theta_step
                r_out_row = Rv + T[i, :]
                for j in range(ncols):
                    x = r_out_row[j] * np.cos(angles[j])
                    z = r_out_row[j] * np.sin(angles[j])
                    vertices.append([x, y, z])
                    colors.append([float(gray[i, j])] * 3)
                    normals.append([0, 1, 0])
                row_angles_list.append(angles.astype(np.float64))
                row_radii_list.append(r_out_row.astype(np.float64))

            panel_rows_angles.append(row_angles_list)
            panel_rows_radii.append(row_radii_list)
            panel_rows_y.append(row_y)

            # stitch outer shell
            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base + i * ncols + j
                    b = base + i * ncols + (j + 1)
                    c = base + (i + 1) * ncols + j
                    d = base + (i + 1) * ncols + (j + 1)
                    indices.append([a, b, c])
                    indices.append([b, d, c])

            top_outer_ring.extend([base + 0 * ncols + j for j in range(ncols - 1, -1, -1)])
            bot_outer_ring.extend([base + (nrows - 1) * ncols + j for j in range(ncols - 1, -1, -1)])

            # inner (smooth) shell
            base_in = panel_base_index()
            for i in range(nrows):
                v       = i / (nrows - 1)
                y       = H - v * H
                Rv      = (1.0 - v) * Rt + v * Rb
                u_prime = 1.0 - (np.arange(ncols, dtype=np.float64) / float(ncols - 1))
                angles  = p_idx * theta_step + u_prime * theta_step
                x = Rv * np.cos(angles)
                z = Rv * np.sin(angles)
                for j in range(ncols):
                    vertices.append([x[j], y, z[j]])
                    colors.append([0.6, 0.6, 0.6])
                    normals.append([0, -1, 0])

            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base_in + i * ncols + j
                    b = base_in + i * ncols + (j + 1)
                    c = base_in + (i + 1) * ncols + j
                    d = base_in + (i + 1) * ncols + (j + 1)
                    indices.append([a, c, b])
                    indices.append([b, c, d])

            top_inner_ring.extend([base_in + 0 * ncols + j for j in range(ncols - 1, -1, -1)])
            bot_inner_ring.extend([base_in + (nrows - 1) * ncols + j for j in range(ncols - 1, -1, -1)])

        # caps
        if top_outer_ring and top_inner_ring:
            stitch_rings(indices, top_inner_ring, top_outer_ring, outward=True)
        if bot_outer_ring and bot_inner_ring:
            stitch_rings(indices, bot_inner_ring, bot_outer_ring, outward=False)

        # ---- brims ----
        mid_gray = (0.6, 0.6, 0.6)
        # top brim
        if p.top_brim_height > 0 and p.top_brim_thickness > 0 and any(panels_present):
            r_in  = Rt
            r_out = Rt + p.top_brim_thickness
            y0    = H
            y1    = H + p.top_brim_height

            brim_inner_y0 = build_ring_xyz_at_radius(r_in,  y0, ncols, num_panels, panels_present)
            brim_outer_y0 = build_ring_xyz_at_radius(r_out, y0, ncols, num_panels, panels_present)

            start_inner_y0 = len(vertices)
            for x, y, z in brim_inner_y0:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, 1, 0])
            inner_y0_idx = list(range(start_inner_y0, start_inner_y0 + len(brim_inner_y0)))
            stitch_rings(indices, top_outer_ring, inner_y0_idx, outward=True)

            start_outer_y0 = len(vertices)
            for x, y, z in brim_outer_y0:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, 1, 0])
            outer_y0_idx = list(range(start_outer_y0, start_outer_y0 + len(brim_outer_y0)))
            stitch_rings(indices, inner_y0_idx, outer_y0_idx, outward=True)

            brim_inner_y1 = [[x, y1, z] for x, _, z in brim_inner_y0]
            brim_outer_y1 = [[x, y1, z] for x, _, z in brim_outer_y0]
            stitch_wall(vertices, colors, normals, indices, brim_inner_y0, brim_inner_y1, color=mid_gray, up=True)
            stitch_wall(vertices, colors, normals, indices, brim_outer_y0, brim_outer_y1, color=mid_gray, up=True)

            start_inner_y1 = len(vertices)
            for x, y, z in brim_inner_y1:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, 1, 0])
            inner_y1_idx = list(range(start_inner_y1, start_inner_y1 + len(brim_inner_y1)))

            start_outer_y1 = len(vertices)
            for x, y, z in brim_outer_y1:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, 1, 0])
            outer_y1_idx = list(range(start_outer_y1, start_outer_y1 + len(brim_outer_y1)))
            stitch_rings(indices, inner_y1_idx, outer_y1_idx, outward=True)

            # ---- sprocket teeth on top brim ----
            if p.sprocket_enabled and p.sprocket_teeth > 0:
                self._build_sprocket(
                    vertices, colors, normals, indices,
                    r_inner=r_in, r_outer=r_out,
                    y_base=y1,
                    tooth_h=p.sprocket_tooth_height,
                    tooth_w_frac=p.sprocket_tooth_width,
                    n_teeth=p.sprocket_teeth,
                    color=mid_gray,
                )

        # bottom brim
        if p.bottom_brim_height > 0 and p.bottom_brim_thickness > 0 and any(panels_present):
            r_in  = Rb
            r_out = Rb + p.bottom_brim_thickness
            y1    = 0.0
            y0    = -p.bottom_brim_height

            brim_inner_y1 = build_ring_xyz_at_radius(r_in,  y1, ncols, num_panels, panels_present)
            brim_outer_y1 = build_ring_xyz_at_radius(r_out, y1, ncols, num_panels, panels_present)

            start_inner_y1 = len(vertices)
            for x, y, z in brim_inner_y1:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, -1, 0])
            inner_y1_idx = list(range(start_inner_y1, start_inner_y1 + len(brim_inner_y1)))
            stitch_rings(indices, inner_y1_idx, bot_outer_ring, outward=False)

            start_outer_y1 = len(vertices)
            for x, y, z in brim_outer_y1:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, -1, 0])
            outer_y1_idx = list(range(start_outer_y1, start_outer_y1 + len(brim_outer_y1)))
            stitch_rings(indices, inner_y1_idx, outer_y1_idx, outward=False)

            brim_inner_y0 = [[x, y0, z] for x, _, z in brim_inner_y1]
            brim_outer_y0 = [[x, y0, z] for x, _, z in brim_outer_y1]
            stitch_wall(vertices, colors, normals, indices, brim_inner_y0, brim_inner_y1, color=mid_gray, up=False)
            stitch_wall(vertices, colors, normals, indices, brim_outer_y0, brim_outer_y1, color=mid_gray, up=False)

            start_inner_y0 = len(vertices)
            for x, y, z in brim_inner_y0:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, -1, 0])
            inner_y0_idx = list(range(start_inner_y0, start_inner_y0 + len(brim_inner_y0)))

            start_outer_y0 = len(vertices)
            for x, y, z in brim_outer_y0:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, -1, 0])
            outer_y0_idx = list(range(start_outer_y0, start_outer_y0 + len(brim_outer_y0)))
            stitch_rings(indices, inner_y0_idx, outer_y0_idx, outward=False)

        # ---- frames (FIXED: inner face samples outer shell; no artificial clearance gap) ----
        if p.frame_width > 0 and p.frame_thickness > 0:
            steps = nrows
            for k in range(num_panels):
                left_idx  = k % num_panels
                right_idx = (k + 1) % num_panels

                left_angles_rows  = panel_rows_angles[left_idx]
                left_radii_rows   = panel_rows_radii[left_idx]
                left_y            = panel_rows_y[left_idx]

                right_angles_rows = panel_rows_angles[right_idx]
                right_radii_rows  = panel_rows_radii[right_idx]
                right_y           = panel_rows_y[right_idx]

                theta_boundary = (k + 1) * theta_step
                half_dw   = (p.frame_width / 2.0) / max(Rt, 1e-6)
                theta_left  = theta_boundary - half_dw
                theta_right = theta_boundary + half_dw

                y_bot = -p.bottom_brim_height if p.bottom_brim_height > 0 else 0.0
                y_top = H + p.top_brim_height  if p.top_brim_height  > 0 else H
                ys = np.linspace(y_bot, y_top, steps + 2, dtype=np.float64)
                ys_clamped = np.clip(ys, 0.0, H)

                pillar_base = len(vertices)

                for y_val in ys_clamped:
                    v  = 0.0 if H <= 0 else (H - y_val) / H
                    Rv = (1.0 - v) * Rt + v * Rb
                    # baseline outer radius at this height (no lithophane relief)
                    baseline_outer = Rv + t_min

                    # LEFT side: sample outer-shell surface directly (flush, no gap)
                    if left_y.size > 0:
                        li  = int(np.argmin(np.abs(left_y - y_val)))
                        la  = np.asarray(left_angles_rows[li]).ravel()
                        lr  = np.asarray(left_radii_rows[li]).ravel()
                        if la.size == 0 or lr.size == 0:
                            r_in_left = baseline_outer
                        else:
                            idx_la = np.argsort(la)
                            la_inc, lr_inc = la[idx_la], lr[idx_la]
                            theta0_c   = np.clip(theta_left, la_inc.min(), la_inc.max())
                            r_in_left  = float(np.interp(theta0_c, la_inc, lr_inc))
                    else:
                        r_in_left = baseline_outer

                    # RIGHT side
                    if right_y.size > 0:
                        ri  = int(np.argmin(np.abs(right_y - y_val)))
                        ra  = np.asarray(right_angles_rows[ri]).ravel()
                        rr  = np.asarray(right_radii_rows[ri]).ravel()
                        if ra.size == 0 or rr.size == 0:
                            r_in_right = baseline_outer
                        else:
                            idx_ra = np.argsort(ra)
                            ra_inc, rr_inc = ra[idx_ra], rr[idx_ra]
                            theta1_c   = np.clip(theta_right, ra_inc.min(), ra_inc.max())
                            r_in_right = float(np.interp(theta1_c, ra_inc, rr_inc))
                    else:
                        r_in_right = baseline_outer

                    # Outer face: baseline_outer + frame_thickness
                    r_out_common = max(
                        baseline_outer + p.frame_thickness,
                        r_in_left  + 1e-3,
                        r_in_right + 1e-3,
                    )

                    xl = r_in_left  * np.cos(theta_left)
                    zl = r_in_left  * np.sin(theta_left)
                    xr = r_in_right * np.cos(theta_right)
                    zr = r_in_right * np.sin(theta_right)
                    xlo = r_out_common * np.cos(theta_left)
                    zlo = r_out_common * np.sin(theta_left)
                    xro = r_out_common * np.cos(theta_right)
                    zro = r_out_common * np.sin(theta_right)

                    for pt in [(xl, y_val, zl), (xr, y_val, zr),
                               (xlo, y_val, zlo), (xro, y_val, zro)]:
                        vertices.append(list(pt))
                        colors.append([0.55, 0.55, 0.55])
                        normals.append([0, 1, 0])

                n_steps = len(ys_clamped)
                for s in range(n_steps - 1):
                    # 4 verts per step: 0=in-left,1=in-right,2=out-left,3=out-right
                    b0 = pillar_base + s * 4
                    b1 = pillar_base + (s + 1) * 4
                    # inner face (faces inward)
                    indices.append([b0+0, b1+0, b0+1])
                    indices.append([b1+0, b1+1, b0+1])
                    # outer face
                    indices.append([b0+2, b0+3, b1+2])
                    indices.append([b1+2, b0+3, b1+3])
                    # left side wall
                    indices.append([b0+0, b0+2, b1+0])
                    indices.append([b1+0, b0+2, b1+2])
                    # right side wall
                    indices.append([b0+1, b1+1, b0+3])
                    indices.append([b1+1, b1+3, b0+3])

                # top cap
                st = pillar_base + (n_steps - 1) * 4
                indices.append([st+0, st+2, st+1])
                indices.append([st+1, st+2, st+3])
                # bottom cap
                sb = pillar_base
                indices.append([sb+0, sb+1, sb+2])
                indices.append([sb+1, sb+3, sb+2])

        # ---- lamp socket ----
        if p.socket_enabled and p.socket_height > 0:
            self._build_socket(
                vertices, colors, normals, indices,
                y_base=H + (p.top_brim_height if p.top_brim_height > 0 else 0.0),
                r_inner=p.socket_inner_diam / 2.0,
                wall=p.socket_wall,
                height=p.socket_height,
                color=mid_gray,
            )

        V = np.array(vertices,  dtype=np.float64)
        I = np.array(indices,   dtype=np.int32)
        N = np.array(normals,   dtype=np.float64)
        C = np.array(colors,    dtype=np.float64)
        return V, I, N, C

    # ------------------------------------------------------------------
    # sprocket teeth
    # ------------------------------------------------------------------
    def _build_sprocket(
        self, vertices, colors, normals, indices,
        r_inner, r_outer, y_base, tooth_h, tooth_w_frac, n_teeth, color,
    ):
        """Add rectangular sprocket teeth on top of the brim top face."""
        arc_step  = 2.0 * np.pi / n_teeth
        tooth_arc = arc_step * float(np.clip(tooth_w_frac, 0.05, 0.95))
        half_arc  = tooth_arc / 2.0
        y_top = y_base + tooth_h

        for t in range(n_teeth):
            a_center = t * arc_step
            a0, a1   = a_center - half_arc, a_center + half_arc

            # 8 corners: bottom inner/outer × left/right,  top inner/outer × left/right
            pts_bot = [
                (r_inner * np.cos(a0), y_base, r_inner * np.sin(a0)),  # 0 bot-in-left
                (r_outer * np.cos(a0), y_base, r_outer * np.sin(a0)),  # 1 bot-out-left
                (r_inner * np.cos(a1), y_base, r_inner * np.sin(a1)),  # 2 bot-in-right
                (r_outer * np.cos(a1), y_base, r_outer * np.sin(a1)),  # 3 bot-out-right
            ]
            pts_top = [
                (r_inner * np.cos(a0), y_top,  r_inner * np.sin(a0)),  # 4 top-in-left
                (r_outer * np.cos(a0), y_top,  r_outer * np.sin(a0)),  # 5 top-out-left
                (r_inner * np.cos(a1), y_top,  r_inner * np.sin(a1)),  # 6 top-in-right
                (r_outer * np.cos(a1), y_top,  r_outer * np.sin(a1)),  # 7 top-out-right
            ]

            base = len(vertices)
            for px, py, pz in pts_bot + pts_top:
                vertices.append([px, py, pz])
                colors.append(list(color))
                normals.append([0.0, 1.0, 0.0])

            # faces (CCW winding outward)
            # top face (y+)
            indices.append([base+4, base+5, base+6])
            indices.append([base+5, base+7, base+6])
            # bottom (y-) — connects to brim top, no need to close unless exporting
            # outer wall
            indices.append([base+1, base+5, base+3])
            indices.append([base+5, base+7, base+3])
            # inner wall
            indices.append([base+0, base+2, base+4])
            indices.append([base+4, base+2, base+6])
            # left wall
            indices.append([base+0, base+4, base+1])
            indices.append([base+4, base+5, base+1])
            # right wall
            indices.append([base+2, base+3, base+6])
            indices.append([base+6, base+3, base+7])

    # ------------------------------------------------------------------
    # lamp socket
    # ------------------------------------------------------------------
    def _build_socket(
        self, vertices, colors, normals, indices,
        y_base, r_inner, wall, height, color, n_seg=64,
    ):
        """Hollow cylinder that slides over the bulb fitting."""
        r_outer = r_inner + wall
        y_top   = y_base + height
        angles  = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)

        def ring(r, y, nrm_y):
            start = len(vertices)
            for a in angles:
                vertices.append([r * np.cos(a), y, r * np.sin(a)])
                colors.append(list(color))
                normals.append([np.cos(a), nrm_y, np.sin(a)])
            return list(range(start, start + n_seg))

        ri_bot = ring(r_inner, y_base, -1.0)
        ro_bot = ring(r_outer, y_base,  1.0)
        ri_top = ring(r_inner, y_top,  -1.0)
        ro_top = ring(r_outer, y_top,   1.0)

        for i in range(n_seg):
            j = (i + 1) % n_seg
            # outer wall
            indices.append([ro_bot[i], ro_top[i], ro_bot[j]])
            indices.append([ro_top[i], ro_top[j], ro_bot[j]])
            # inner wall
            indices.append([ri_bot[i], ri_bot[j], ri_top[i]])
            indices.append([ri_top[i], ri_bot[j], ri_top[j]])
            # top annulus
            indices.append([ri_top[i], ri_top[j], ro_top[i]])
            indices.append([ro_top[i], ri_top[j], ro_top[j]])
            # bottom annulus
            indices.append([ri_bot[i], ro_bot[i], ri_bot[j]])
            indices.append([ro_bot[i], ro_bot[j], ri_bot[j]])

    # ------------------------------------------------------------------
    # sphere builder (unchanged)
    # ------------------------------------------------------------------
    def _build_sphere(self, imgs):
        p = self.p
        R  = float(p.top_diam) / 2.0
        nrows, ncols, num_panels = p.nrows, p.ncols, p.num_panels
        theta_step = 2.0 * np.pi / num_panels
        t_min = float(p.min_thickness)
        t_max = float(p.max_thickness)

        vertices, colors, normals, indices = [], [], [], []

        for p_idx, img in enumerate(imgs):
            if img is None:
                continue
            data = np.asarray(img.resize((ncols, nrows)), dtype=np.float32) / 255.0
            T    = t_min + (1.0 - data) * (t_max - t_min)
            gray = 1.0 - ((T - t_min) / max(t_max - t_min, 1e-6))

            base = len(vertices)
            for i in range(nrows):
                phi = np.pi * i / (nrows - 1)
                for j in range(ncols):
                    u = j / (ncols - 1)
                    theta = p_idx * theta_step + u * theta_step
                    r = R + T[i, j]
                    x = r * np.sin(phi) * np.cos(theta)
                    y = r * np.cos(phi)
                    z = r * np.sin(phi) * np.sin(theta)
                    vertices.append([x, y, z])
                    colors.append([float(gray[i, j])] * 3)
                    normals.append([np.sin(phi)*np.cos(theta),
                                    np.cos(phi),
                                    np.sin(phi)*np.sin(theta)])

            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base + i * ncols + j
                    b = base + i * ncols + (j + 1)
                    c = base + (i + 1) * ncols + j
                    d = base + (i + 1) * ncols + (j + 1)
                    indices.append([a, b, c])
                    indices.append([b, d, c])

        V = np.array(vertices, dtype=np.float64)
        I = np.array(indices,  dtype=np.int32)
        N = np.array(normals,  dtype=np.float64)
        C = np.array(colors,   dtype=np.float64)
        return V, I, N, C

    # ------------------------------------------------------------------
    # flat builder (unchanged)
    # ------------------------------------------------------------------
    def _build_flat(self, imgs):
        p = self.p
        W  = float(p.bottom_diam)
        H  = float(p.height)
        t_min = float(p.min_thickness)
        t_max = float(p.max_thickness)
        nrows, ncols = p.nrows, p.ncols
        panel_w = W / max(p.num_panels, 1)

        vertices, colors, normals, indices = [], [], [], []

        for p_idx, img in enumerate(imgs):
            if img is None:
                continue
            data = np.asarray(img.resize((ncols, nrows)), dtype=np.float32) / 255.0
            T    = t_min + (1.0 - data) * (t_max - t_min)
            gray = 1.0 - ((T - t_min) / max(t_max - t_min, 1e-6))

            x_off = p_idx * panel_w
            base  = len(vertices)

            for i in range(nrows):
                for j in range(ncols):
                    x = x_off + j / (ncols - 1) * panel_w
                    y = i / (nrows - 1) * H
                    z = T[i, j]
                    vertices.append([x, y, z])
                    colors.append([float(gray[i, j])] * 3)
                    normals.append([0, 0, 1])

            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base + i * ncols + j
                    b = base + i * ncols + (j + 1)
                    c = base + (i + 1) * ncols + j
                    d = base + (i + 1) * ncols + (j + 1)
                    indices.append([a, b, c])
                    indices.append([b, d, c])

        V = np.array(vertices, dtype=np.float64)
        I = np.array(indices,  dtype=np.int32)
        N = np.array(normals,  dtype=np.float64)
        C = np.array(colors,   dtype=np.float64)
        return V, I, N, C
