import struct
import numpy as np


def _coerce_vertices(vertices) -> np.ndarray:
    arr = np.asarray(vertices)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _coerce_indices(indices) -> np.ndarray:
    arr = np.asarray(indices)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return np.ascontiguousarray(arr, dtype=np.int64)
    if arr.ndim == 1 and arr.size > 0:
        rows = []
        for idx, tri in enumerate(arr):
            try:
                a, b, c = tri
            except Exception:
                raise ValueError(f"indices[{idx}] is not a 3-tuple: {tri!r}")
            rows.append([int(a), int(b), int(c)])
        return np.ascontiguousarray(np.array(rows, dtype=np.int64), dtype=np.int64)
    raise ValueError(f"indices must be shape (M,3) or list of triplets, got {arr.shape}")


def _compute_face_normals(verts: np.ndarray, inds: np.ndarray) -> np.ndarray:
    v1 = verts[inds[:, 0]]
    v2 = verts[inds[:, 1]]
    v3 = verts[inds[:, 2]]
    n  = np.cross(v2 - v1, v3 - v1).astype(np.float32)
    L  = np.linalg.norm(n, axis=1)
    L[L == 0] = 1.0
    n  = (n.T / L).T
    return np.ascontiguousarray(n, dtype=np.float32)


def save_binary_stl(
    path: str,
    vertices,
    indices,
    header_text: str = "Lithophane (mm)",
    smooth_inside: bool = True,
) -> None:
    """Binary STL writer (80-byte header, uint32 count, 50 B/facet)."""
    verts = _coerce_vertices(vertices)
    inds  = _coerce_indices(indices)

    normals = _compute_face_normals(verts, inds)
    if smooth_inside:
        normals = -normals
        inds    = inds.copy()
        inds[:, [1, 2]] = inds[:, [2, 1]]

    tri_count = int(inds.shape[0])

    with open(path, "wb") as f:
        hdr = (header_text or "Binary STL").encode("ascii", "ignore")[:80].ljust(80, b"\0")
        f.write(hdr)
        f.write(struct.pack("<I", tri_count))
        for i in range(tri_count):
            a, b, c = int(inds[i, 0]), int(inds[i, 1]), int(inds[i, 2])
            nx, ny, nz = float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2])
            f.write(struct.pack("<fff", nx, ny, nz))
            f.write(struct.pack("<fff", *verts[a]))
            f.write(struct.pack("<fff", *verts[b]))
            f.write(struct.pack("<fff", *verts[c]))
            f.write(struct.pack("<H", 0))
