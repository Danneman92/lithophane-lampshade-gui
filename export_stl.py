"""Binary STL export — supports single mesh or multiple named meshes."""
import os
import struct
import numpy as np


def _write_binary_stl(path: str, vertices: np.ndarray,
                      indices: np.ndarray, header: str = "Lithophane (mm)") -> None:
    """Write a single binary STL file."""
    verts = np.asarray(vertices, dtype=np.float32)
    inds  = np.asarray(indices,  dtype=np.int32)
    triangles = verts[inds]  # (F, 3, 3)
    e0 = triangles[:, 1] - triangles[:, 0]
    e1 = triangles[:, 2] - triangles[:, 0]
    face_normals = np.cross(e0, e1).astype(np.float32)
    mag = np.linalg.norm(face_normals, axis=1, keepdims=True)
    mag = np.where(mag < 1e-12, 1.0, mag)
    face_normals /= mag
    n_faces = len(inds)
    header_bytes = header.encode("utf-8")[:80].ljust(80, b"\x00")
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(struct.pack("<I", n_faces))
        for k in range(n_faces):
            nx, ny, nz = face_normals[k]
            f.write(struct.pack("<fff", nx, ny, nz))
            for corner in triangles[k]:
                f.write(struct.pack("<fff", *corner))
            f.write(struct.pack("<H", 0))


def save_binary_stl(path: str, vertices, indices,
                   header_text: str = "Lithophane (mm)",
                   smooth_inside: bool = False) -> None:
    """Save shade mesh as a single STL."""
    _write_binary_stl(path, vertices, indices, header_text)


def save_all_stl(base_path: str,
                 shade_mesh,
                 socket_mesh=None) -> dict:
    """Save shade and (optionally) socket as separate STL files.

    Returns a dict of {label: path} for the files written.
    """
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".stl"

    written = {}

    # Shade
    shade_path = root + "_shade" + ext
    V, I = shade_mesh[0], shade_mesh[1]
    _write_binary_stl(shade_path, V, I, "Lithophane Shade (mm)")
    written["shade"] = shade_path

    # Socket / interface
    if socket_mesh is not None:
        Vs, Is = socket_mesh[0], socket_mesh[1]
        if len(Is) > 0:
            sock_path = root + "_socket" + ext
            _write_binary_stl(sock_path, Vs, Is, "Lamp Socket Adapter (mm)")
            written["socket"] = sock_path

    return written
