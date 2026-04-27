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
    socket_enabled: bool = False
    socket_inner_diam: float = 32.5
    socket_wall: float = 3.5
    socket_height: float = 25.0
    socket_lip_height: float = 3.5
    socket_lip_overhang: float = 1.5
    spokes_enabled: bool = False
    spoke_count: int = 4
    spoke_width: float = 6.0
    spoke_thickness: float = 6.0
    top_brim_fillet: float = 4.9
    top_brim_fillet_steps: int = 6


class LithophaneBuilder:
    def __init__(self, params: BuildParams):
        self.p = params

    def build(self, imgs):
        if self.p.shade_type == "Sphere":
            return self._build_sphere(imgs)
        if self.p.shade_type == "Flat":
            return self._build_flat(imgs)
        return self._build_cylinder(imgs)

    def _build_cylinder(self, imgs):
        p = self.p
        H  = float(p.height)
        Rt = float(p.top_diam)    / 2.0
        Rb = float(p.bottom_diam) / 2.0
        nrows, ncols, num_panels = p.nrows, p.ncols, p.num_panels
        theta_step = 2.0 * np.pi / num_panels
        t_min = float(p.min_thickness)
        t_max = float(p.max_thickness)

        vertices, colors, normals, indices = [], [], [], []
        top_outer_ring, top_inner_ring = [], []
        bot_outer_ring, bot_inner_ring = [], []
        panels_present = [img is not None for img in imgs]

        panel_rows_angles: List[List[np.ndarray]] = []
        panel_rows_radii:  List[List[np.ndarray]] = []
        panel_rows_y:      List[np.ndarray]       = []

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

            base = len(vertices)
            for i in range(nrows):
                v  = i / (nrows - 1)
                y  = H - v * H
                row_y[i] = y
                Rv = (1.0 - v) * Rt + v * Rb
                u_prime   = 1.0 - (np.arange(ncols, dtype=np.float64) / float(ncols - 1))
                angles    = p_idx * theta_step + u_prime * theta_step
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

            base_in = len(vertices)
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

        if top_outer_ring and top_inner_ring:
            stitch_rings(indices, top_inner_ring, top_outer_ring, outward=True)
        if bot_outer_ring and bot_inner_ring:
            stitch_rings(indices, bot_outer_ring, bot_outer_ring, outward=False)

        mid_gray = (0.6, 0.6, 0.6)

        # ------------------------------------------------------------------
        # Top brim  (fully self-contained solid ring)
        # ------------------------------------------------------------------
        if p.top_brim_height > 0 and p.top_brim_thickness > 0 and any(panels_present):
            r_in     = Rt
            r_out    = Rt + p.top_brim_thickness
            y0       = H
            y1       = H + p.top_brim_height
            fillet_r = float(p.top_brim_fillet)
            fillet_r = max(fillet_r, 0.0)
            fillet_n = max(int(p.top_brim_fillet_steps), 2) if fillet_r > 0 else 0
            n_pts = ncols * num_panels

            def _brim_ring(r, y, normal_type='up'):
                s = len(vertices)
                angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                for ang in angles:
                    x = r * np.cos(ang)
                    z = r * np.sin(ang)
                    vertices.append([x, y, z])
                    colors.append(list(mid_gray))
                    if normal_type == 'outward':
                        normals.append([np.cos(ang), 0.0, np.sin(ang)])
                    elif normal_type == 'inward':
                        normals.append([-np.cos(ang), 0.0, -np.sin(ang)])
                    elif normal_type == 'down':
                        normals.append([0.0, -1.0, 0.0])
                    else:
                        normals.append([0.0, 1.0, 0.0])
                return list(range(s, s + n_pts))

            def _stitch(inner_idx, outer_idx, outward=True):
                n = len(inner_idx)
                for ii in range(n):
                    jj = (ii + 1) % n
                    a, b = inner_idx[ii], inner_idx[jj]
                    c, d = outer_idx[ii], outer_idx[jj]
                    if outward:
                        indices.append([a, c, b])
                        indices.append([b, c, d])
                    else:
                        indices.append([a, b, c])
                        indices.append([b, d, c])

            inner_y0_idx = _brim_ring(r_in, y0, normal_type='up')
            stitch_rings(indices, inner_y0_idx, top_outer_ring, outward=True)

            inner_top_idx = _brim_ring(r_in,  y1, normal_type='up')
            outer_top_idx = _brim_ring(r_out, y1, normal_type='up')

            if fillet_r > 0:
                fbase_idx = _brim_ring(r_out - fillet_r, y0, normal_type='down')
                _stitch(inner_y0_idx, fbase_idx, outward=True)

                prev_idx = fbase_idx
                for t in np.linspace(0.0, np.pi / 2.0, fillet_n + 1)[1:]:
                    r_arc    = (r_out - fillet_r) + fillet_r * np.sin(t)
                    y_arc    = (y0   + fillet_r) - fillet_r * np.cos(t)
                    curr_idx = _brim_ring(r_arc, y_arc, normal_type='outward')
                    _stitch(prev_idx, curr_idx, outward=False)
                    prev_idx = curr_idx

                _stitch(prev_idx, outer_top_idx, outward=False)
            else:
                outer_y0_idx = _brim_ring(r_out, y0, normal_type='down')
                _stitch(inner_y0_idx, outer_y0_idx, outward=True)
                _stitch(outer_y0_idx, outer_top_idx, outward=False)

            _stitch(inner_y0_idx, inner_top_idx, outward=True)
            _stitch(inner_top_idx, outer_top_idx, outward=False)

        # ------------------------------------------------------------------
        # Bottom brim  (fully self-contained solid ring)
        # ------------------------------------------------------------------
        if p.bottom_brim_height > 0 and p.bottom_brim_thickness > 0 and any(panels_present):
            r_in  = Rb
            r_out = Rb + p.bottom_brim_thickness
            y1, y0 = 0.0, -p.bottom_brim_height

            brim_inner_y1 = build_ring_xyz_at_radius(r_in,  y1, ncols, num_panels, panels_present)
            brim_outer_y1 = build_ring_xyz_at_radius(r_out, y1, ncols, num_panels, panels_present)

            si1 = len(vertices)
            for x, y, z in brim_inner_y1:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, -1, 0])
            inner_y1_idx = list(range(si1, si1 + len(brim_inner_y1)))
            stitch_rings(indices, inner_y1_idx, bot_outer_ring, outward=False)

            so1 = len(vertices)
            for x, y, z in brim_outer_y1:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, -1, 0])
            outer_y1_idx = list(range(so1, so1 + len(brim_outer_y1)))
            stitch_rings(indices, inner_y1_idx, outer_y1_idx, outward=False)

            brim_inner_y0 = [[x, y0, z] for x, _, z in brim_inner_y1]
            brim_outer_y0 = [[x, y0, z] for x, _, z in brim_outer_y1]
            stitch_wall(vertices, colors, normals, indices, brim_inner_y0, brim_inner_y1, color=mid_gray, up=False)
            stitch_wall(vertices, colors, normals, indices, brim_outer_y0, brim_outer_y1, color=mid_gray, up=False)

            si0 = len(vertices)
            for x, y, z in brim_inner_y0:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, -1, 0])
            inner_y0_idx = list(range(si0, si0 + len(brim_inner_y0)))

            so0 = len(vertices)
            for x, y, z in brim_outer_y0:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, -1, 0])
            outer_y0_idx = list(range(so0, so0 + len(brim_outer_y0)))
            stitch_rings(indices, inner_y0_idx, outer_y0_idx, outward=False)

        # ---- frames / pillars ---------------------------------------------------
        # Pillars run y=0..H (shade body only) so they never intersect brim faces.
        # Separate fully-enclosed filler blocks bridge each pillar into the brims.
        if p.frame_width > 0 and p.frame_thickness > 0:
            steps = nrows

            has_top_brim = p.top_brim_height > 0 and p.top_brim_thickness > 0
            has_bot_brim = p.bottom_brim_height > 0 and p.bottom_brim_thickness > 0

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
                half_dw        = (p.frame_width / 2.0) / max(Rt, 1e-6)
                theta_left     = theta_boundary - half_dw
                theta_right    = theta_boundary + half_dw

                # --- main pillar: y=0 to y=H ---
                ys_clamped = np.linspace(0.0, H, steps + 1, dtype=np.float64)

                pillar_base = len(vertices)
                pillar_radii_by_level = []

                for y_val in ys_clamped:
                    v  = 0.0 if H <= 0 else (H - y_val) / H
                    Rv = (1.0 - v) * Rt + v * Rb
                    baseline_outer = Rv + t_min

                    if left_y.size > 0:
                        li = int(np.argmin(np.abs(left_y - y_val)))
                        la = np.asarray(left_angles_rows[li]).ravel()
                        lr = np.asarray(left_radii_rows[li]).ravel()
                        if la.size and lr.size:
                            idx_la = np.argsort(la)
                            la_s, lr_s = la[idx_la], lr[idx_la]
                            r_in_left = float(np.interp(
                                np.clip(theta_left, la_s.min(), la_s.max()), la_s, lr_s))
                        else:
                            r_in_left = baseline_outer
                    else:
                        r_in_left = baseline_outer

                    if right_y.size > 0:
                        ri = int(np.argmin(np.abs(right_y - y_val)))
                        ra = np.asarray(right_angles_rows[ri]).ravel()
                        rr = np.asarray(right_radii_rows[ri]).ravel()
                        if ra.size and rr.size:
                            idx_ra = np.argsort(ra)
                            ra_s, rr_s = ra[idx_ra], rr[idx_ra]
                            r_in_right = float(np.interp(
                                np.clip(theta_right, ra_s.min(), ra_s.max()), ra_s, rr_s))
                        else:
                            r_in_right = baseline_outer
                    else:
                        r_in_right = baseline_outer

                    r_out_common = max(
                        baseline_outer + p.frame_thickness,
                        r_in_left  + 1e-3,
                        r_in_right + 1e-3,
                    )
                    pillar_radii_by_level.append((r_in_left, r_in_right, r_out_common))

                    for idx, (r, theta) in enumerate([(r_in_left, theta_left), (r_in_right, theta_right),
                                         (r_out_common, theta_left), (r_out_common, theta_right)]):
                        vx = r * np.cos(theta)
                        vz = r * np.sin(theta)
                        vertices.append([vx, y_val, vz])
                        colors.append([0.55, 0.55, 0.55])
                        if idx < 2:
                            normals.append([-np.cos(theta), 0, -np.sin(theta)])
                        else:
                            normals.append([np.cos(theta), 0, np.sin(theta)])

                n_levels = len(ys_clamped)
                for level_idx in range(n_levels - 1):
                    dr_out = pillar_radii_by_level[level_idx + 1][2] - pillar_radii_by_level[level_idx][2]
                    dy = ys_clamped[level_idx + 1] - ys_clamped[level_idx]
                    slope = dr_out / (dy + 1e-6)
                    for outer_idx in [2, 3]:
                        vert_idx = pillar_base + level_idx * 4 + outer_idx
                        vpos = vertices[vert_idx]
                        theta_v = np.arctan2(vpos[2], vpos[0])
                        adjusted_n = np.array([np.cos(theta_v), 0.2 * slope, np.sin(theta_v)])
                        adjusted_n = adjusted_n / (np.linalg.norm(adjusted_n) + 1e-6)
                        normals[vert_idx] = adjusted_n.tolist()

                n_steps = len(ys_clamped)
                for s in range(n_steps - 1):
                    b0 = pillar_base + s * 4
                    b1 = pillar_base + (s + 1) * 4
                    indices.append([b0+0, b1+0, b0+1]); indices.append([b1+0, b1+1, b0+1])
                    indices.append([b0+2, b0+3, b1+2]); indices.append([b1+2, b0+3, b1+3])
                    indices.append([b0+0, b0+2, b1+0]); indices.append([b1+0, b0+2, b1+2])
                    indices.append([b0+1, b1+1, b0+3]); indices.append([b1+1, b1+3, b0+3])

                # bottom cap of main pillar - normal DOWN
                sb = pillar_base
                cap_bot = len(vertices)
                for ci in range(4):
                    vertices.append(list(vertices[sb + ci]))
                    colors.append([0.55, 0.55, 0.55])
                    normals.append([0.0, -1.0, 0.0])
                indices.append([cap_bot+0, cap_bot+1, cap_bot+2])
                indices.append([cap_bot+1, cap_bot+3, cap_bot+2])

                # top cap of main pillar - normal UP
                et = pillar_base + (n_steps - 1) * 4
                cap_top = len(vertices)
                for ci in range(4):
                    vertices.append(list(vertices[et + ci]))
                    colors.append([0.55, 0.55, 0.55])
                    normals.append([0.0, 1.0, 0.0])
                indices.append([cap_top+0, cap_top+2, cap_top+1])
                indices.append([cap_top+1, cap_top+2, cap_top+3])

                # ------------------------------------------------------------------
                # Top brim filler block: a standalone closed box sitting on top of
                # the brim, same angular width as the pillar, spanning the full
                # brim thickness (r_in=Rt to r_out=Rt+brim_thickness), from y=H
                # to y=H+brim_height.  It never touches the brim ring geometry.
                # ------------------------------------------------------------------
                if has_top_brim:
                    tb_r_in  = Rt
                    tb_r_out = Rt + float(p.top_brim_thickness)
                    tb_y0    = H
                    tb_y1    = H + float(p.top_brim_height)

                    # 8 corners of the box:
                    # 0: inner-left  bottom  1: inner-right bottom
                    # 2: outer-left  bottom  3: outer-right bottom
                    # 4: inner-left  top     5: inner-right top
                    # 6: outer-left  top     7: outer-right top
                    tb_corners = [
                        [tb_r_in  * np.cos(theta_left),  tb_y0, tb_r_in  * np.sin(theta_left)],
                        [tb_r_in  * np.cos(theta_right), tb_y0, tb_r_in  * np.sin(theta_right)],
                        [tb_r_out * np.cos(theta_left),  tb_y0, tb_r_out * np.sin(theta_left)],
                        [tb_r_out * np.cos(theta_right), tb_y0, tb_r_out * np.sin(theta_right)],
                        [tb_r_in  * np.cos(theta_left),  tb_y1, tb_r_in  * np.sin(theta_left)],
                        [tb_r_in  * np.cos(theta_right), tb_y1, tb_r_in  * np.sin(theta_right)],
                        [tb_r_out * np.cos(theta_left),  tb_y1, tb_r_out * np.sin(theta_left)],
                        [tb_r_out * np.cos(theta_right), tb_y1, tb_r_out * np.sin(theta_right)],
                    ]
                    tb_base = len(vertices)
                    for cx, cy, cz in tb_corners:
                        vertices.append([cx, cy, cz])
                        colors.append([0.55, 0.55, 0.55])
                        normals.append([0.0, 1.0, 0.0])  # placeholder, all faces use dedicated verts below

                    def _tb_face(a, b, c, d):
                        """CCW quad (two tris) from corner indices, normal outward by winding."""
                        indices.append([tb_base+a, tb_base+b, tb_base+c])
                        indices.append([tb_base+b, tb_base+d, tb_base+c])

                    # bottom face (normal DOWN): 0,1,2,3
                    _tb_face(0, 2, 1, 3)   # swap to flip normal down
                    # top face (normal UP): 4,5,6,7
                    _tb_face(4, 5, 6, 7)
                    # inner face (normal inward): 0,1,4,5
                    _tb_face(1, 0, 5, 4)
                    # outer face (normal outward): 2,3,6,7
                    _tb_face(2, 6, 3, 7)
                    # left face: 0,2,4,6
                    _tb_face(0, 4, 2, 6)
                    # right face: 1,3,5,7
                    _tb_face(3, 1, 7, 5)

                # ------------------------------------------------------------------
                # Bottom brim filler block: same idea, from y=0 down to y=-brim_height
                # ------------------------------------------------------------------
                if has_bot_brim:
                    bb_r_in  = Rb
                    bb_r_out = Rb + float(p.bottom_brim_thickness)
                    bb_y1    = 0.0
                    bb_y0    = -float(p.bottom_brim_height)

                    bb_corners = [
                        [bb_r_in  * np.cos(theta_left),  bb_y0, bb_r_in  * np.sin(theta_left)],
                        [bb_r_in  * np.cos(theta_right), bb_y0, bb_r_in  * np.sin(theta_right)],
                        [bb_r_out * np.cos(theta_left),  bb_y0, bb_r_out * np.sin(theta_left)],
                        [bb_r_out * np.cos(theta_right), bb_y0, bb_r_out * np.sin(theta_right)],
                        [bb_r_in  * np.cos(theta_left),  bb_y1, bb_r_in  * np.sin(theta_left)],
                        [bb_r_in  * np.cos(theta_right), bb_y1, bb_r_in  * np.sin(theta_right)],
                        [bb_r_out * np.cos(theta_left),  bb_y1, bb_r_out * np.sin(theta_left)],
                        [bb_r_out * np.cos(theta_right), bb_y1, bb_r_out * np.sin(theta_right)],
                    ]
                    bb_base = len(vertices)
                    for cx, cy, cz in bb_corners:
                        vertices.append([cx, cy, cz])
                        colors.append([0.55, 0.55, 0.55])
                        normals.append([0.0, -1.0, 0.0])

                    def _bb_face(a, b, c, d):
                        indices.append([bb_base+a, bb_base+b, bb_base+c])
                        indices.append([bb_base+b, bb_base+d, bb_base+c])

                    # bottom face (normal DOWN): 0,1,2,3
                    _bb_face(0, 1, 2, 3)
                    # top face (normal UP): 4,5,6,7
                    _bb_face(4, 6, 5, 7)
                    # inner face: 0,1,4,5
                    _bb_face(1, 0, 5, 4)
                    # outer face: 2,3,6,7
                    _bb_face(2, 6, 3, 7)
                    # left face: 0,2,4,6
                    _bb_face(0, 4, 2, 6)
                    # right face: 1,3,5,7
                    _bb_face(3, 1, 7, 5)

        # ---- lamp socket + spokes -----------------------------------------------
        y_bed = -float(p.bottom_brim_height) if p.bottom_brim_height > 0 else 0.0

        if p.socket_enabled and p.socket_height > 0:
            r_sock_in  = p.socket_inner_diam / 2.0
            r_sock_out = r_sock_in + p.socket_wall
            self._build_socket(
                vertices, colors, normals, indices,
                y_base=y_bed,
                y_top=float(p.socket_height),
                r_inner=r_sock_in,
                r_outer=r_sock_out,
                lip_height=p.socket_lip_height,
                lip_overhang=p.socket_lip_overhang,
                color=mid_gray,
            )

            if p.spokes_enabled and p.spoke_count > 0:
                self._build_spokes(
                    vertices, colors, normals, indices,
                    y_bot=y_bed,
                    y_top=y_bed + float(p.spoke_thickness),
                    r_hub=r_sock_out,
                    r_rim=Rb,
                    n_spokes=p.spoke_count,
                    spoke_w=p.spoke_width,
                    color=mid_gray,
                )

        V = np.array(vertices, dtype=np.float64)
        I = np.array(indices,  dtype=np.int32)
        N = np.array(normals,  dtype=np.float64)
        C = np.array(colors,   dtype=np.float64)
        return V, I, N, C

    def _build_socket(
        self, vertices, colors, normals, indices,
        y_base, y_top, r_inner, r_outer,
        lip_height, lip_overhang, color, n_seg=64,
    ):
        angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        r_lip  = max(r_inner - lip_overhang, 1.0)
        y_lip  = y_top - lip_height

        def add_ring(r, y, nrm_fn):
            start = len(vertices)
            for a in angles:
                vertices.append([r * np.cos(a), y, r * np.sin(a)])
                colors.append(list(color))
                normals.append(list(nrm_fn(a)))
            return list(range(start, start + n_seg))

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
            indices.append([a, b, c])
            indices.append([a, c, d])

        for i in range(n_seg):
            j = (i + 1) % n_seg
            quad(ro_base[i], ro_top[i],  ro_top[j],  ro_base[j])
            quad(ri_lip[j],  ri_base[j], ri_base[i], ri_lip[i])
            quad(ro_base[i], ri_base[i], ri_base[j], ro_base[j])
            quad(rl_lip[j],  rl_top[j],  rl_top[i],  rl_lip[i])
            quad(ri_lip[i],  rl_lip[i],  rl_lip[j],  ri_lip[j])
            quad(rl_top[i],  ri_top[i],  ri_top[j],  rl_top[j])
            quad(ri_top[i],  ro_top[i],  ro_top[j],  ri_top[j])

    def _build_spokes(
        self, vertices, colors, normals, indices,
        y_bot, y_top, r_hub, r_rim, n_spokes, spoke_w, color,
    ):
        angle_step = 2.0 * np.pi / n_spokes
        half_w     = spoke_w / 2.0

        for k in range(n_spokes):
            a_ctr = k * angle_step
            tx = -np.sin(a_ctr)
            tz =  np.cos(a_ctr)
            rx =  np.cos(a_ctr)
            rz =  np.sin(a_ctr)

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

            def f(a, b, c):
                indices.append([base + a, base + b, base + c])

            f(4, 6, 5); f(5, 6, 7)
            f(0, 1, 2); f(1, 3, 2)
            f(0, 4, 1); f(4, 5, 1)
            f(2, 3, 6); f(3, 7, 6)
            f(0, 2, 4); f(4, 2, 6)
            f(1, 5, 3); f(5, 7, 3)

    def _build_sphere(self, imgs):
        p = self.p
        R  = float(p.top_diam) / 2.0
        nrows, ncols, num_panels = p.nrows, p.ncols, p.num_panels
        theta_step = 2.0 * np.pi / num_panels
        t_min, t_max = float(p.min_thickness), float(p.max_thickness)
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
                    normals.append([np.sin(phi)*np.cos(theta), np.cos(phi),
                                    np.sin(phi)*np.sin(theta)])
            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = base + i*ncols + j;     b = base + i*ncols + (j+1)
                    c = base + (i+1)*ncols + j; d = base + (i+1)*ncols + (j+1)
                    indices.append([a, b, c]); indices.append([b, d, c])
        return (np.array(vertices, dtype=np.float64), np.array(indices, dtype=np.int32),
                np.array(normals,  dtype=np.float64), np.array(colors,  dtype=np.float64))

    def _build_flat(self, imgs):
        p = self.p
        W, H = float(p.bottom_diam), float(p.height)
        t_min, t_max = float(p.min_thickness), float(p.max_thickness)
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
                    a = base + i*ncols + j;     b = base + i*ncols + (j+1)
                    c = base + (i+1)*ncols + j; d = base + (i+1)*ncols + (j+1)
                    indices.append([a, b, c]); indices.append([b, d, c])
        return (np.array(vertices, dtype=np.float64), np.array(indices, dtype=np.int32),
                np.array(normals,  dtype=np.float64), np.array(colors,  dtype=np.float64))
