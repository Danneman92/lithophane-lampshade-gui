import struct
import numpy as np

def _coerce_vertices(vertices) -> np.ndarray:
    """
    Return contiguous (N,3) float32 array; raise if not 3D points.
    """
    arr = np.asarray(vertices)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)

def _coerce_indices(indices) -> np.ndarray:
    """
    Accepts:
      - (M,3) numeric array
      - iterable of 3-tuples/lists (ragged), i.e. shape (M,) dtype=object
    Returns contiguous (M,3) int64 array.
    """
    arr = np.asarray(indices)
    # Case 1: already (M,3)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return np.ascontiguousarray(arr, dtype=np.int64)
    # Case 2: ragged list/array of triplets
    if arr.ndim == 1 and arr.size > 0 and hasattr(arr, "__len__") and len(arr) == 3:
        return np.ascontiguousarray(np.array(list(arr), dtype=np.int64).reshape(-1, 3), dtype=np.int64)
    raise ValueError(f"indices must be (M,3) or list of 3-tuples; got {arr.shape} dtype={arr.dtype}")

def _compute_face_normals(verts: np.ndarray, inds: np.ndarray) -> np.ndarray:
    """
    Per-triangle normals via cross product (right-hand rule), normalized. [2]
    """
    v1 = verts[inds[:, 0]]
    v2 = verts[inds[:, 1]]
    v3 = verts[inds[:, 2]]
    n = np.cross(v2 - v1, v3 - v1).astype(np.float32)
    L = np.linalg.norm(n, axis=1)
    L[L == 0] = 1.0
    n = (n.T / L).T
    return np.ascontiguousarray(n, dtype=np.float32)

def save_binary_stl(path: str,
                    vertices,
                    indices,
                    header_text: str = "Lithophane (mm)",
                    smooth_inside: bool = True) -> None:
    """
    Binary STL writer (80B header, uint32 count, 50B/facet). [2][3]
    If smooth_inside=True, negate facet normals and swap v2/v3 to reverse winding so the inside is front-lit. [2][4]
    """
    verts = _coerce_vertices(vertices)      # (N,3) float32 [2]
    inds  = _coerce_indices(indices)        # (M,3) int64   [2]

    normals = _compute_face_normals(verts, inds)  # (M,3) float32 [2]

    if smooth_inside:
        normals = -normals                              # flip lighting inward [2]
        inds = inds.copy()
        inds[:, [1, 2]] = inds[:, [2, 1]]              # reverse winding (right-hand rule) [2][5]

    tri_count = int(inds.shape)                     # Python int, not tuple/np scalar [2]

    with open(path, "wb") as f:
        # 80-byte header (avoid starting with 'solid' to prevent ASCII mis-detection). [2]
        hdr = (header_text or "Binary STL").encode("ascii", "ignore")[:80].ljust(80, b"\0")
        f.write(hdr)

        # uint32 triangle count (little-endian). [3][2]
        f.write(struct.pack("<I", tri_count))

        # Each facet: normal(3f) + v1(3f) + v2(3f) + v3(3f) + uint16 attribute (0). [2]
        for i in range(tri_count):
            nx, ny, nz = float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2])
            a0, a1, a2 = int(inds[i, 0]), int(inds[i, 1]), int(inds[i, 2])
            x0, y0, z0 = map(float, verts[a0])
            x1, y1, z1 = map(float, verts[a1])
            x2, y2, z2 = map(float, verts[a2])
            f.write(struct.pack("<12fH",
                                nx, ny, nz,
                                x0, y0, z0,
                                x1, y1, z1,
                                x2, y2, z2,
                                0))
