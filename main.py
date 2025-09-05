#!/usr/bin/env python3
"""
Lithophane Lampshade GUI Application
- Load up to 4 photos with live thumbnails
- Min/Max thickness mapping (white=min, black=max)
- 3D preview with unlit grayscale based on thickness (no lighting)
- Geometry oriented Y-up (height along Y), relief radial in XZ
- NEW: Top/Bottom diameter inputs for conical (frustum) shapes
"""
import sys
import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSpinBox, QGridLayout, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from glwidget import GLWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lithophane Lampshade Creator")
        self.setGeometry(100, 100, 1200, 900)

        # State
        self.image_paths = [None] * 4
        self.images = [None] * 4
        self.img_labels = []
        self.preview_labels = []

        # Parameters
        self.height = 150               # mm
        self.top_diam = 160             # mm (top diameter)
        self.bottom_diam = 160          # mm (bottom diameter)
        self.min_thickness = 0.30       # mm (white)
        self.max_thickness = 3.00       # mm (black)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout()

        # Left controls
        controls_box = QVBoxLayout()

        grid = QGridLayout()
        for i in range(4):
            btn = QPushButton(f"Load Photo {i+1}")
            btn.clicked.connect(lambda _, idx=i: self.select_image(idx))
            name = QLabel("No file")
            thumb = QLabel()
            thumb.setFixedSize(100, 100)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet("border: 1px solid #888; background:#222;")
            grid.addWidget(btn, i, 0)
            grid.addWidget(name, i, 1)
            grid.addWidget(thumb, i, 2)
            self.img_labels.append(name)
            self.preview_labels.append(thumb)

        controls_box.addLayout(grid)

        # Height
        controls_box.addWidget(QLabel("Height"))
        sp_h = QSpinBox()
        sp_h.setRange(50, 400)
        sp_h.setValue(self.height)
        sp_h.valueChanged.connect(lambda v: setattr(self, "height", v))
        controls_box.addWidget(sp_h)

        # Top diameter
        controls_box.addWidget(QLabel("Top Diameter"))
        sp_dt = QSpinBox()
        sp_dt.setRange(40, 600)
        sp_dt.setValue(self.top_diam)
        sp_dt.valueChanged.connect(lambda v: setattr(self, "top_diam", v))
        controls_box.addWidget(sp_dt)

        # Bottom diameter
        controls_box.addWidget(QLabel("Bottom Diameter"))
        sp_db = QSpinBox()
        sp_db.setRange(40, 600)
        sp_db.setValue(self.bottom_diam)
        sp_db.valueChanged.connect(lambda v: setattr(self, "bottom_diam", v))
        controls_box.addWidget(sp_db)

        # Min thickness
        controls_box.addWidget(QLabel("Min Thickness"))
        sp_min = QDoubleSpinBox()
        sp_min.setRange(0.1, 10.0)
        sp_min.setDecimals(2)
        sp_min.setSingleStep(0.05)
        sp_min.setValue(self.min_thickness)
        sp_min.valueChanged.connect(lambda v: setattr(self, "min_thickness", v))
        controls_box.addWidget(sp_min)

        # Max thickness
        controls_box.addWidget(QLabel("Max Thickness"))
        sp_max = QDoubleSpinBox()
        sp_max.setRange(0.1, 10.0)
        sp_max.setDecimals(2)
        sp_max.setSingleStep(0.05)
        sp_max.setValue(self.max_thickness)
        sp_max.valueChanged.connect(lambda v: setattr(self, "max_thickness", v))
        controls_box.addWidget(sp_max)

        # Generate
        gen = QPushButton("Generate Lampshade")
        gen.setStyleSheet("QPushButton{background:#2e7d32;color:white;font-weight:bold;}")
        gen.clicked.connect(self.generate_model)
        controls_box.addWidget(gen)

        left_widget = QWidget()
        left_widget.setLayout(controls_box)
        root.addWidget(left_widget)

        # Right: view buttons + GL widget
        self.gl_widget = GLWidget()

        view_bar = QHBoxLayout()
        for name in ["Iso", "Front", "Back", "Left", "Right", "Top", "Bottom"]:
            b = QPushButton(name)
            b.clicked.connect(lambda _, n=name: self.gl_widget.set_view(n))
            view_bar.addWidget(b)

        right_col = QVBoxLayout()
        vw = QWidget()
        vw.setLayout(view_bar)
        right_col.addWidget(vw)
        right_col.addWidget(self.gl_widget, stretch=1)

        right_widget = QWidget()
        right_widget.setLayout(right_col)
        root.addWidget(right_widget, stretch=1)

        central.setLayout(root)
        self.setCentralWidget(central)

    def select_image(self, idx: int):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        self.image_paths[idx] = path
        filename = path.split("/")[-1]
        self.img_labels[idx].setText(filename if len(filename) <= 24 else filename[:21] + "...")
        img = Image.open(path)
        self.images[idx] = img

        # thumbnail
        thumb = img.copy()
        thumb.thumbnail((100, 100))
        try:
            qimg = ImageQt.ImageQt(thumb)
            pix = QPixmap.fromImage(qimg)
            self.preview_labels[idx].setPixmap(pix)
        except Exception:
            thumb.save("_preview_tmp.png")
            self.preview_labels[idx].setPixmap(QPixmap("_preview_tmp.png"))

    def generate_model(self):
        # Load grayscale images
        imgs = [Image.open(p).convert("L") if p else None for p in self.image_paths]
        if not any(imgs):
            return

        # Parameters
        num_panels = 4
        theta_step = 2.0 * np.pi / num_panels
        H = float(self.height)
        Rt = float(self.top_diam) / 2.0          # top radius
        Rb = float(self.bottom_diam) / 2.0       # bottom radius
        t_min = float(self.min_thickness)
        t_max = float(self.max_thickness)

        # Resolution of each panel grid
        nrows, ncols = 120, 160  # rows along height (Y), cols around circumference

        vertices = []
        indices = []
        normals = []
        colors = []
        vi = 0

        for panel_idx, img in enumerate(imgs):
            if img is None:
                continue

            # Prepare thickness map (white=min, black=max)
            img_resized = img.resize((ncols, nrows))
            data = np.asarray(img_resized, dtype=np.float32) / 255.0  # 0..1, white=1 thin
            thickness = t_min + (1.0 - data) * (t_max - t_min)        # thickness in mm

            # Grayscale from thickness: white for thin, black for thick
            norm_t = (thickness - t_min) / max(t_max - t_min, 1e-6)
            gray = 1.0 - norm_t

            # Build vertices grid for this panel, Y-up, radial in XZ
            panel_verts = np.zeros((nrows, ncols, 3), dtype=np.float32)
            panel_cols  = np.zeros((nrows, ncols, 3), dtype=np.float32)

            for i in range(nrows):
                # y from top(H) to bottom(0) so object is upright
                v = i / (nrows - 1)                # 0..1 top→bottom
                y = H - v * H
                # Linear radius interpolation for a right circular frustum
                Rv = (1.0 - v) * Rt + v * Rb       # base radius at this height
                for j in range(ncols):
                    # Reverse angular progression so images read left→right outside
                    u = 1.0 - (j / (ncols - 1))    # 1→0 around the panel
                    angle = panel_idx * theta_step + u * theta_step
                    r = Rv - thickness[i, j]       # inner surface relief
                    x = r * np.cos(angle)
                    z = r * np.sin(angle)
                    panel_verts[i, j] = [x, y, z]
                    g = gray[i, j]
                    panel_cols[i, j] = [g, g, g]

            # Compute normals (central differences)
            panel_norms = np.zeros_like(panel_verts)
            for i in range(nrows):
                for j in range(ncols):
                    i0 = max(i - 1, 0); i1 = min(i + 1, nrows - 1)
                    j0 = max(j - 1, 0); j1 = min(j + 1, ncols - 1)
                    du = panel_verts[i, j1] - panel_verts[i, j0]  # around circumference
                    dv = panel_verts[i1, j] - panel_verts[i0, j]  # along height
                    n = np.cross(du, dv)
                    norm = np.linalg.norm(n)
                    panel_norms[i, j] = (n / norm) if norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

            # Append flattened arrays
            vertices.extend(panel_verts.reshape(-1, 3).tolist())
            colors.extend(panel_cols.reshape(-1, 3).tolist())
            normals.extend(panel_norms.reshape(-1, 3).tolist())

            # Indices
            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    a = vi + i * ncols + j
                    b = vi + i * ncols + (j + 1)
                    c = vi + (i + 1) * ncols + j
                    d = vi + (i + 1) * ncols + (j + 1)
                    indices.append([a, b, c])
                    indices.append([b, d, c])

            vi += nrows * ncols

        # Optional: add top/bottom rings (neutral mid-gray)
        ring_steps = 180
        ring_gray = 0.5
        for y, Rring in ((H, Rt), (0.0, Rb)):
            for k in range(ring_steps):
                ang = 2.0 * np.pi * (k / ring_steps)
                x = (Rring + t_max) * np.cos(ang)
                z = (Rring + t_max) * np.sin(ang)
                vertices.append([x, y, z])
                normals.append([0.0, 1.0 if y == H else -1.0, 0.0])
                colors.append([ring_gray, ring_gray, ring_gray])

        # Send to viewer (pass colors)
        self.gl_widget.update_geometry(
            np.asarray(vertices, dtype=np.float32),
            np.asarray(indices, dtype=np.uint32),
            np.asarray(normals, dtype=np.float32),
            np.asarray(colors, dtype=np.float32),
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
