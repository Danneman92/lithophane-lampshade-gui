from dataclasses import dataclass
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

class LithophaneBuilder:
    def __init__(self, params: BuildParams):
        self.p = params

    def build(self, imgs: List[Optional["Image.Image"]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.p
        H = float(p.height)
        Rt = float(p.top_diam)/2.0
        Rb = float(p.bottom_diam)/2.0
        Rmid = 0.5*(Rt+Rb)
        t_min = float(p.min_thickness); t_max = float(p.max_thickness)
        nrows, ncols, num_panels = p.nrows, p.ncols, p.num_panels
        theta_step = 2.0*np.pi/num_panels

        vertices: List[List[float]] = []
        colors:   List[List[float]] = []
        normals:  List[List[float]] = []
        indices:  List[List[int]] = []

        top_outer_ring: List[int] = []
        top_inner_ring: List[int] = []
        bot_outer_ring: List[int] = []
        bot_inner_ring: List[int] = []
        boundary_col_idx: List[List[int]] = []
        panels_present = [img is not None for img in imgs]

        def panel_base_index() -> int:
            return len(vertices)

        # Body: outer + inner shells
        for p_idx, img in enumerate(imgs):
            if img is None: 
                continue
            data = np.asarray(img.resize((ncols, nrows)), dtype=np.float32)/255.0
            T = t_min + (1.0 - data) * (t_max - t_min)
            gray = 1.0 - ((T - t_min)/max(t_max - t_min, 1e-6))

            # Outer
            base = panel_base_index()
            for i in range(nrows):
                v = i/(nrows-1)
                y = H - v*H
                Rv = (1.0 - v)*Rt + v*Rb
                for j in range(ncols):
                    u_prime = 1.0 - (j/(ncols-1))
                    angle = p_idx*theta_step + u_prime*theta_step
                    x = Rv*np.cos(angle); z = Rv*np.sin(angle)
                    vertices.append([x,y,z]); colors.append([float(gray[i,j])]*3); normals.append([0,1,0])
            # quads
            for i in range(nrows-1):
                for j in range(ncols-1):
                    a = base + i*ncols + j
                    b = base + i*ncols + (j+1)
                    c = base + (i+1)*ncols + j
                    d = base + (i+1)*ncols + (j+1)
                    indices.append([a,b,c]); indices.append([b,d,c])
            # rings (increasing angle => reverse j)
            top_outer_ring.extend([base + 0*ncols + j for j in range(ncols-1, -1, -1)])
            bot_outer_ring.extend([base + (nrows-1)*ncols + j for j in range(ncols-1, -1, -1)])
            boundary_col_idx.append([base + i*ncols + (ncols-1) for i in range(nrows)])

            # Inner (blue-ish so it’s visible)
            base_in = panel_base_index()
            for i in range(nrows):
                v = i/(nrows-1)
                y = H - v*H
                Rv = (1.0 - v)*Rt + v*Rb
                for j in range(ncols):
                    u_prime = 1.0 - (j/(ncols-1))
                    angle = p_idx*theta_step + u_prime*theta_step
                    rin = Rv - T[i,j]
                    x = rin*np.cos(angle); z = rin*np.sin(angle)
                    vertices.append([x,y,z]); colors.append([0.3,0.3,0.8]); normals.append([0,-1,0])
            for i in range(nrows-1):
                for j in range(ncols-1):
                    a = base_in + i*ncols + j
                    b = base_in + i*ncols + (j+1)
                    c = base_in + (i+1)*ncols + j
                    d = base_in + (i+1)*ncols + (j+1)
                    indices.append([a,c,b]); indices.append([b,c,d])
            top_inner_ring.extend([base_in + 0*ncols + j for j in range(ncols-1, -1, -1)])
            bot_inner_ring.extend([base_in + (nrows-1)*ncols + j for j in range(ncols-1, -1, -1)])

        # Caps (annuli)
        if top_outer_ring and top_inner_ring:
            stitch_rings(indices, top_inner_ring, top_outer_ring, outward=True)   # top annulus [24]
        if bot_outer_ring and bot_inner_ring:
            stitch_rings(indices, bot_inner_ring, bot_outer_ring, outward=False)  # bottom annulus [24]

        mid_gray = (0.6,0.6,0.6)

        # Solid TOP BRIM
        if p.top_brim_height > 0 and p.top_brim_thickness > 0 and any(panels_present):
            r_in, r_out = Rt, Rt + p.top_brim_thickness
            y0, y1 = H, H + p.top_brim_height
            brim_inner_y0 = build_ring_xyz_at_radius(r_in, y0, ncols, num_panels, panels_present)
            brim_outer_y0 = build_ring_xyz_at_radius(r_out, y0, ncols, num_panels, panels_present)

            # add inner y0 and stitch body->brim inner seam
            start_inner_y0 = len(vertices)
            for x,y,z in brim_inner_y0:
                vertices.append([x,y,z]); colors.append(list(mid_gray)); normals.append([0,1,0])
            inner_y0_idx = list(range(start_inner_y0, start_inner_y0 + len(brim_inner_y0)))
            stitch_rings(indices, top_outer_ring, inner_y0_idx, outward=True)  # continuity seam [24]

            # add outer y0, bottom cap
            start_outer_y0 = len(vertices)
            for x,y,z in brim_outer_y0:
                vertices.append([x,y,z]); colors.append(list(mid_gray)); normals.append([0,1,0])
            outer_y0_idx = list(range(start_outer_y0, start_outer_y0 + len(brim_outer_y0)))
            stitch_rings(indices, inner_y0_idx, outer_y0_idx, outward=True)    # bottom cap [24]

            # walls and top cap
            brim_inner_y1 = [[x,y1,z] for x,_,z in brim_inner_y0]
            brim_outer_y1 = [[x,y1,z] for x,_,z in brim_outer_y0]
            stitch_wall(vertices, colors, normals, indices, brim_inner_y0, brim_inner_y1, color=mid_gray, up=True)  # inner wall [23]
            stitch_wall(vertices, colors, normals, indices, brim_outer_y0, brim_outer_y1, color=mid_gray, up=True)  # outer wall [23]
            # add top ring verts
            start_inner_y1 = len(vertices)
            for x,y,z in brim_inner_y1:
                vertices.append([x,y,z]); colors.append(list(mid_gray)); normals.append([0,1,0])
            inner_y1_idx = list(range(start_inner_y1, start_inner_y1 + len(brim_inner_y1)))
            start_outer_y1 = len(vertices)
            for x,y,z in brim_outer_y1:
                vertices.append([x,y,z]); colors.append(list(mid_gray)); normals.append([0,1,0])
            outer_y1_idx = list(range(start_outer_y1, start_outer_y1 + len(brim_outer_y1)))
            stitch_rings(indices, inner_y1_idx, outer_y1_idx, outward=True)     # top cap [24]

        # Solid BOTTOM BRIM
        if p.bottom_brim_height > 0 and p.bottom_brim_thickness > 0 and any(panels_present):
            r_in, r_out = Rb, Rb + p.bottom_brim_thickness
            y1, y0 = 0.0, -p.bottom_brim_height
            brim_inner_y1 = build_ring_xyz_at_radius(r_in, y1, ncols, num_panels, panels_present)
            brim_outer_y1 = build_ring_xyz_at_radius(r_out, y1, ncols, num_panels, panels_present)
            # add inner y1 and stitch to body bottom ring
            start_inner_y1 = len(vertices)
            for x,y,z in brim_inner_y1:
                vertices.append([x,y,z]); colors.append(list(mid_gray)); normals.append([0,-1,0])
            inner_y1_idx = list(range(start_inner_y1, start_inner_y1 + len(brim_inner_y1)))
            stitch_rings(indices, inner_y1_idx, bot_outer_ring, outward=False)  # continuity seam [24]
            # add outer y1 and top cap
            start_outer_y1 = len(vertices)
            for x,y,z in brim_outer_y1:
                vertices.append([x,y,z]); colors.append(list(mid_gray)); normals.append([0,-1,0])
            outer_y1_idx = list(range(start_outer_y1, start_outer_y1 + len(brim_outer_y1)))
            stitch_rings(indices, inner_y1_idx, outer_y1_idx, outward=False)    # top cap [24]
            # walls down
            brim_inner_y0 = [[x,y0,z] for x,_,z in brim_inner_y1]
            brim_outer_y0 = [[x,y0,z] for x,_,z in brim_outer_y1]
            stitch_wall(vertices, colors, normals, indices, brim_inner_y0, brim_inner_y1, color=mid_gray, up=False) # inner wall [23]
            stitch_wall(vertices, colors, normals, indices, brim_outer_y0, brim_outer_y1, color=mid_gray, up=False) # outer wall [23]
            # bottom cap
            start_inner_y0 = len(vertices)
            for x,y,z in brim_inner_y0:
                vertices.append([x,y,z]); colors.append(list(mid_gray)); normals.append([0,-1,0])
            inner_y0_idx = list(range(start_inner_y0, start_inner_y0 + len(brim_inner_y0)))
            start_outer_y0 = len(vertices)
            for x,y,z in brim_outer_y0:
                vertices.append([x,y,z]); colors.append(list(mid_gray)); normals.append([0,-1,0])
            outer_y0_idx = list(range(start_outer_y0, start_outer_y0 + len(brim_outer_y0)))
            stitch_rings(indices, inner_y0_idx, outer_y0_idx, outward=False)     # bottom cap [24]

        # Optional vertical frames
        if p.frame_width > 0 and p.frame_thickness > 0 and boundary_col_idx:
            delta_theta = max(p.frame_width / max(Rmid, 1e-6), 1e-6)
            for k, outer_edge in enumerate(boundary_col_idx):
                center_theta = k * (2.0*np.pi/num_panels)
                theta_out = center_theta + 0.5*delta_theta
                outer_xyz = []
                for v_idx in outer_edge:
                    x0,y0,z0 = vertices[v_idx]
                    v = 1.0 - (y0 / H) if H > 1e-9 else 0.0
                    Rv = (1.0 - v)*Rt + v*Rb
                    r = Rv + p.frame_thickness
                    x = r*np.cos(theta_out); z = r*np.sin(theta_out)
                    outer_xyz.append([x,y0,z])
                # strip between existing edge and new column
                start_outer = len(vertices)
                for x,y,z in outer_xyz:
                    vertices.append([x,y,z]); colors.append([0.6,0.6,0.6]); normals.append([0,0,1])
                for i in range(len(outer_edge)-1):
                    a = outer_edge[i]; b = outer_edge[i+1]
                    c = start_outer + i; d = start_outer + i + 1
                    indices.append([a,c,b]); indices.append([c,d,b])

        V = np.asarray(vertices, dtype=np.float32)
        I = np.asarray(indices, dtype=np.uint32)
        N = np.asarray(normals, dtype=np.float32)
        C = np.asarray(colors, dtype=np.float32)
        return V, I, N, C
