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
    # Lamp socket adapter (sits inside the shade, bottom at y=0)
    socket_enabled: bool = False
    socket_inner_diam: float = 26.0   # bore to slip over bulb fitting (mm)
    socket_wall: float = 2.5          # cylinder wall thickness (mm)
    socket_height: float = 60.0       # how tall the tube is inside the shade
    socket_lip_height: float = 4.0    # height of the inward lip at the TOP
    socket_lip_overhang: float = 4.0  # how far the lip narrows the bore inward
    # Spokes connecting socket to inner shade wall
    spokes_enabled: bool = False
    spoke_count: int = 4              # number of spokes
    spoke_width: float = 4.0          # spoke width in mm (constant, tangential)
    spoke_thickness: float = 2.0      # spoke thickness (mm, vertical)


class LithophaneBuilder:
    def __init__(self, params: BuildParams):
        self.p = params

    def build(self, imgs):
        if self.p.shade_type == "Sphere":
            return self._build_sphere(imgs)
        if self.p.shade_type == "Flat":
            return self._build_flat(imgs)
        return self._build_cylinder(imgs)

    # ------------------------------------------------------------------
    # cylinder builder
    # ------------------------------------------------------------------
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

        # ---- outer + inner shells ------------------------------------------------
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

            # inner shell
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

        # caps
        if top_outer_ring and top_inner_ring:
            stitch_rings(indices, top_inner_ring, top_outer_ring, outward=True)
        if bot_outer_ring and bot_inner_ring:
            stitch_rings(indices, bot_inner_ring, bot_outer_ring, outward=False)

        # ---- brims ---------------------------------------------------------------
        mid_gray = (0.6, 0.6, 0.6)

        if p.top_brim_height > 0 and p.top_brim_thickness > 0 and any(panels_present):
            r_in  = Rt
            r_out = Rt + p.top_brim_thickness
            y0, y1 = H, H + p.top_brim_height

            brim_inner_y0 = build_ring_xyz_at_radius(r_in,  y0, ncols, num_panels, panels_present)
            brim_outer_y0 = build_ring_xyz_at_radius(r_out, y0, ncols, num_panels, panels_present)

            si0 = len(vertices)
            for x, y, z in brim_inner_y0:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, 1, 0])
            inner_y0_idx = list(range(si0, si0 + len(brim_inner_y0)))
            stitch_rings(indices, top_outer_ring, inner_y0_idx, outward=True)

            so0 = len(vertices)
            for x, y, z in brim_outer_y0:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, 1, 0])
            outer_y0_idx = list(range(so0, so0 + len(brim_outer_y0)))
            stitch_rings(indices, inner_y0_idx, outer_y0_idx, outward=True)

            brim_inner_y1 = [[x, y1, z] for x, _, z in brim_inner_y0]
            brim_outer_y1 = [[x, y1, z] for x, _, z in brim_outer_y0]
            stitch_wall(vertices, colors, normals, indices, brim_inner_y0, brim_inner_y1, color=mid_gray, up=True)
            stitch_wall(vertices, colors, normals, indices, brim_outer_y0, brim_outer_y1, color=mid_gray, up=True)

            si1 = len(vertices)
            for x, y, z in brim_inner_y1:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, 1, 0])
            inner_y1_idx = list(range(si1, si1 + len(brim_inner_y1)))

            so1 = len(vertices)
            for x, y, z in brim_outer_y1:
                vertices.append([x, y, z]); colors.append(list(mid_gray)); normals.append([0, 1, 0])
            outer_y1_idx = list(range(so1, so1 + len(brim_outer_y1)))
            stitch_rings(indices, inner_y1_idx, outer_y1_idx, outward=True)

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

        # ---- frames / pillars (flush against lithophane) -------------------------
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
                half_dw     = (p.frame_width / 2.0) / max(Rt, 1e-6)
                theta_left  = theta_boundary - half_dw
                theta_right = theta_boundary + half_dw

                y_bot = -p.bottom_brim_height if p.bottom_brim_height > 0 else 0.0
                y_top = H + p.top_brim_height  if p.top_brim_height  > 0 else H
                ys_clamped = np.clip(
                    np.linspace(y_bot, y_top, steps + 2, dtype=np.float64), 0.0, H
                )

                pillar_base = len(vertices)

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

                    for r, theta in [(r_in_left, theta_left), (r_in_right, theta_right),
                                     (r_out_common, theta_left), (r_out_common, theta_right)]:
                        vertices.append([r * np.cos(theta), y_val, r * np.sin(theta)])
                        colors.append([0.55, 0.55, 0.55])
                        normals.append([0, 1, 0])

                n_steps = len(ys_clamped)
                for s in range(n_steps - 1):
                    b0 = pillar_base + s * 4
                    b1 = pillar_base + (s + 1) * 4
                    indices.append([b0+0, b1+0, b0+1]); indices.append([b1+0, b1+1, b0+1])
                    indices.append([b0+2, b0+3, b1+2]); indices.append([b1+2, b0+3, b1+3])
                    indices.append([b0+0, b0+2, b1+0]); indices.append([b1+0, b0+2, b1+2])
                    indices.append([b0+1, b1+1, b0+3]); indices.append([b1+1, b1+3, b0+3])

                st = pillar_base + (n_steps - 1) * 4
                indices.append([st+0, st+2, st+1]); indices.append([st+1, st+2, st+3])
                sb = pillar_base
                indices.append([sb+0, sb+1, sb+2]); indices.append([sb+1, sb+3, sb+2])

        # ---- lamp socket + spokes ------------------------------------------------
        if p.socket_enabled and p.socket_height > 0:
            r_sock_in  = p.socket_inner_diam / 2.0
            r_sock_out = r_sock_in + p.socket_wall
            self._build_socket(
                vertices, colors, normals, indices,
                y_base=0.0,
                y_top=float(p.socket_height),
                r_inner=r_sock_in,
                r_outer=r_sock_out,
                lip_height=p.socket_lip_height,
                lip_overhang=p.socket_lip_overhang,
                color=mid_gray,
            )

            if p.spokes_enabled and p.spoke_count > 0:
                r_shade_inner = Rb
                self._build_spokes(
                    vertices, colors, normals, indices,
                    y=0.0,
                    r_hub=r_sock_out,
                    r_rim=r_shade_inner,
                    n_spokes=p.spoke_count,
                    spoke_w=p.spoke_width,
                    spoke_t=p.spoke_thickness,
                    color=mid_gray,
                )

        V = np.array(vertices, dtype=np.float64)
        I = np.array(indices,  dtype=np.int32)
        N = np.array(normals,  dtype=np.float64)
        C = np.array(colors,   dtype=np.float64)
        return V, I, N, C

    # ------------------------------------------------------------------
    # lamp socket
    # Tube: y_base (bottom, open) -> y_top (top).
    # Lip: at the TOP, faces INWARD (narrows the bore like a retaining ring).
    #      When the fitting is pushed up through the bottom of the shade,
    #      the lip catches on the fitting collar and stops it pulling out.
    # ------------------------------------------------------------------
    def _build_socket(
        self, vertices, colors, normals, indices,
        y_base, y_top, r_inner, r_outer,
        lip_height, lip_overhang,
        color, n_seg=64,
    ):
        angles  = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        # Lip sits at the top, protruding INWARD: inner radius shrinks by lip_overhang
        r_lip   = max(r_inner - lip_overhang, 1.0)   # clamped to at least 1 mm bore
        y_lip   = y_top - lip_height                  # bottom of the lip collar

        def add_ring(r, y, nrm_fn):
            start = len(vertices)
            for a in angles:
                vertices.append([r * np.cos(a), y, r * np.sin(a)])
                colors.append(list(color))
                normals.append(list(nrm_fn(a)))
            return list(range(start, start + n_seg))

        radial_out = lambda a: [ np.cos(a), 0.0,  np.sin(a)]
        radial_in  = lambda a: [-np.cos(a), 0.0, -np.sin(a)]
        face_up    = lambda a: [0.0,  1.0, 0.0]
        face_dn    = lambda a: [0.0, -1.0, 0.0]

        # ---- rings from bottom to top ----
        ri_base = add_ring(r_inner, y_base, radial_in)   # inner wall, bottom
        ro_base = add_ring(r_outer, y_base, radial_out)  # outer wall, bottom
        ri_lip  = add_ring(r_inner, y_lip,  radial_in)   # inner wall at lip base
        ro_lip  = add_ring(r_outer, y_lip,  radial_out)  # outer wall at lip base
        # lip collar — inner edge narrows to r_lip
        rl_bot  = add_ring(r_lip,   y_lip,  face_dn)     # lip inner edge, bottom
        rl_top  = add_ring(r_lip,   y_top,  face_up)     # lip inner edge, top
        ri_top  = add_ring(r_inner, y_top,  radial_in)   # inner wall, top
        ro_top  = add_ring(r_outer, y_top,  radial_out)  # outer wall, top

        def quad(a, b, c, d):
            """Two triangles forming a quad, CCW winding."""
            indices.append([a, b, c])
            indices.append([a, c, d])

        for i in range(n_seg):
            j = (i + 1) % n_seg

            # Outer wall: full height y_base -> y_top
            quad(ro_base[i], ro_top[i], ro_top[j], ro_base[j])

            # Inner wall: y_base -> y_lip (below the lip)
            quad(ri_lip[j], ri_base[j], ri_base[i], ri_lip[i])

            # Bottom annulus cap (open bottom of tube)
            quad(ri_base[i], ro_base[i], ro_base[j], ri_base[j])

            # Lip section — inward collar at the top
            # Inner face of lip (the narrowed bore surface): y_lip -> y_top
            quad(rl_bot[j], rl_top[j], rl_top[i], rl_bot[i])
            # Lip bottom annulus (horizontal face at y_lip, from r_lip to r_inner)
            quad(rl_bot[i], ri_lip[i], ri_lip[j], rl_bot[j])
            # Lip top annulus cap (horizontal face at y_top, from r_lip to r_outer)
            quad(ro_top[i], ri_top[i], rl_top[i], ro_top[j])   # outer arc
            quad(ri_top[i], rl_top[i], rl_top[j], ri_top[j])   # inner arc to lip
            # Outer wall section from y_lip to y_top already covered above.
            # Inner wall section from y_lip to y_top (between r_inner and r_lip)
            quad(ri_lip[i], ri_top[i], ri_top[j], ri_lip[j])

    # ------------------------------------------------------------------
    # spokes  — uniform rectangular cross-section all the way along
    # ------------------------------------------------------------------
    def _build_spokes(
        self, vertices, colors, normals, indices,
        y, r_hub, r_rim, n_spokes, spoke_w, spoke_t, color,
    ):
        """
        Flat rectangular spokes radiating from r_hub to r_rim at height y.
        spoke_w is the CONSTANT tangential width in mm at every radius —
        so the same number of mm wide at the hub end and the rim end.
        spoke_t is the vertical thickness in mm.
        """
        y_bot      = y
        y_top      = y + spoke_t
        angle_step = 2.0 * np.pi / n_spokes
        half_w     = spoke_w / 2.0  # half-width in mm (Cartesian, not angular)

        for k in range(n_spokes):
            a_ctr = k * angle_step
            # Unit tangent direction at this spoke centre
            tx = -np.sin(a_ctr)   # tangent x
            tz =  np.cos(a_ctr)   # tangent z
            # Unit radial direction
            rx =  np.cos(a_ctr)
            rz =  np.sin(a_ctr)

            # The four lateral offsets in world-space (±half_w along tangent)
            # Combined with the two radii and two y values -> 8 corners
            def pt(r, side, yy):
                # side = +1 or -1 selects left/right in tangential direction
                cx = r * rx + side * half_w * tx
                cz = r * rz + side * half_w * tz
                return [cx, yy, cz]

            corners = [
                pt(r_hub, -1, y_bot),  # 0 hub-left-bot
                pt(r_hub, +1, y_bot),  # 1 hub-right-bot
                pt(r_rim, -1, y_bot),  # 2 rim-left-bot
                pt(r_rim, +1, y_bot),  # 3 rim-right-bot
                pt(r_hub, -1, y_top),  # 4 hub-left-top
                pt(r_hub, +1, y_top),  # 5 hub-right-top
                pt(r_rim, -1, y_top),  # 6 rim-left-top
                pt(r_rim, +1, y_top),  # 7 rim-right-top
            ]

            base = len(vertices)
            for cx, cy, cz in corners:
                vertices.append([cx, cy, cz])
                colors.append(list(color))
                normals.append([0.0, 1.0, 0.0])

            def f(a, b, c):
                indices.append([base + a, base + b, base + c])

            # top face  (y+)
            f(4, 6, 5); f(5, 6, 7)
            # bottom face (y-)
            f(0, 1, 2); f(1, 3, 2)
            # inner (hub) face
            f(0, 4, 1); f(4, 5, 1)
            # outer (rim) face
            f(2, 3, 6); f(3, 7, 6)
            # left side
            f(0, 2, 4); f(4, 2, 6)
            # right side
            f(1, 5, 3); f(5, 7, 3)

    # ------------------------------------------------------------------
    # sphere builder
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
                    u     = j / (ncols - 1)
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
                    a = base + i * ncols + j;     b = base + i * ncols + (j + 1)
                    c = base + (i + 1) * ncols + j; d = base + (i + 1) * ncols + (j + 1)
                    indices.append([a, b, c]); indices.append([b, d, c])

        return (np.array(vertices, dtype=np.float64), np.array(indices, dtype=np.int32),
                np.array(normals,  dtype=np.float64), np.array(colors,  dtype=np.float64))

    # ------------------------------------------------------------------
    # flat builder
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
                    a = base + i * ncols + j;     b = base + i * ncols + (j + 1)
                    c = base + (i + 1) * ncols + j; d = base + (i + 1) * ncols + (j + 1)
                    indices.append([a, b, c]); indices.append([b, d, c])

        return (np.array(vertices, dtype=np.float64), np.array(indices, dtype=np.int32),
                np.array(normals,  dtype=np.float64), np.array(colors,  dtype=np.float64))
