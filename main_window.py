"""Main window - mirrors every parameter from lithophanemaker.com/Lamp Lithophane.html"""
import dataclasses
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QGridLayout, QDoubleSpinBox, QScrollArea,
    QGroupBox, QFormLayout, QSplitter, QSizePolicy, QMessageBox,
    QSlider, QComboBox, QSpinBox, QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer
from glwidget import GLWidget
from image_utils import load_thumbnail_pixmap
from builder import LithophaneBuilder, BuildParams
from export_stl import save_all_stl, save_binary_stl


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lithophane Lamp Maker")
        self.resize(1350, 980)

        self.panel_count  = 4
        self.image_paths  = [None] * self.panel_count
        self.img_labels   = []
        self.preview_labels = []

        self.params = BuildParams()

        self._shade_mesh  = None
        self._socket_mesh = None

        # debounce timer so spinbox typing doesn't rebuild on every keystroke
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(400)
        self._regen_timer.timeout.connect(self.generate_model)

        self._build_ui()

    # ------------------------------------------------------------------ helpers
    def _dspin(self, val, lo, hi, dec=1, step=1.0, suffix="", tip=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setSingleStep(step)
        w.setValue(val)
        if suffix: w.setSuffix(" " + suffix)
        if tip:    w.setToolTip(tip)
        w.setMinimumWidth(90)
        return w

    def _ispin(self, val, lo, hi, tip=""):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        if tip: w.setToolTip(tip)
        w.setMinimumWidth(90)
        return w

    def _connect(self, widget, attr):
        """Wire any spin/check to params.<attr> and schedule regeneration."""
        def _set(v):
            setattr(self.params, attr, v)
            self._regen_timer.start()
        if isinstance(widget, QCheckBox):
            widget.stateChanged.connect(lambda s: _set(bool(s)))
        elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            widget.valueChanged.connect(_set)
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(_set)
        return widget

    def _row(self, form, label, widget):
        form.addRow(label, widget)
        return widget

    # ------------------------------------------------------------------ groups
    def _grp_lithophane(self):
        """Section 1 - Lithophane Parameters (matches website exactly)"""
        g = QGroupBox("1. Lithophane Parameters")
        f = QFormLayout()

        # Outer Diameter
        self.w_outer_diam = self._dspin(self.params.outer_diameter, 20, 2000, 1, 5, "mm",
            "Outer diameter of the lampshade")
        self._connect(self.w_outer_diam, "outer_diameter")
        f.addRow("Outer Diameter", self.w_outer_diam)

        # Wall Height
        self.w_wall_height = self._dspin(self.params.wall_height, 10, 1000, 1, 5, "mm",
            "Height of the lithophane wall")
        self._connect(self.w_wall_height, "wall_height")
        f.addRow("Wall Height", self.w_wall_height)

        # Wall Thickness (layer thickness range)
        self.w_min_thick = self._dspin(self.params.min_thickness, 0.5, 5, 2, 0.1, "mm",
            "Minimum wall thickness (lightest / most transparent)")
        self._connect(self.w_min_thick, "min_thickness")
        f.addRow("Min Thickness", self.w_min_thick)

        self.w_max_thick = self._dspin(self.params.max_thickness, 0.5, 10, 2, 0.1, "mm",
            "Maximum wall thickness (darkest / most opaque)")
        self._connect(self.w_max_thick, "max_thickness")
        f.addRow("Max Thickness", self.w_max_thick)

        # Number of Sides (panels)
        self.w_num_sides = self._ispin(self.params.num_sides, 1, 12,
            "Number of image panels / sides")
        self._connect(self.w_num_sides, "num_sides")
        self.w_num_sides.valueChanged.connect(self._on_sides_changed)
        f.addRow("Number of Sides", self.w_num_sides)

        # Frame Width
        self.w_frame_width = self._dspin(self.params.frame_width, 0, 30, 1, 0.5, "mm",
            "Width of the solid frame pillar between panels")
        self._connect(self.w_frame_width, "frame_width")
        f.addRow("Frame Width", self.w_frame_width)

        # Inner Diameter (top opening)
        self.w_inner_diam = self._dspin(self.params.inner_diameter, 0, 1000, 1, 5, "mm",
            "Inner diameter at the top of the shade (0 = auto from wall thickness)")
        self._connect(self.w_inner_diam, "inner_diameter")
        f.addRow("Inner Diameter (top)", self.w_inner_diam)

        # Top thickness (collar)
        self.w_top_thickness = self._dspin(self.params.top_thickness, 0, 30, 1, 0.5, "mm",
            "Thickness of the top collar / brim")
        self._connect(self.w_top_thickness, "top_thickness")
        f.addRow("Top Thickness", self.w_top_thickness)

        # Top Height
        self.w_top_height = self._dspin(self.params.top_height, 0, 100, 1, 0.5, "mm",
            "Height of the solid top collar")
        self._connect(self.w_top_height, "top_height")
        f.addRow("Top Height", self.w_top_height)

        # Bottom thickness
        self.w_bot_thickness = self._dspin(self.params.bottom_thickness, 0, 30, 1, 0.5, "mm",
            "Thickness of the bottom ring")
        self._connect(self.w_bot_thickness, "bottom_thickness")
        f.addRow("Bottom Thickness", self.w_bot_thickness)

        # Bottom Height
        self.w_bot_height = self._dspin(self.params.bottom_height, 0, 100, 1, 0.5, "mm",
            "Height of the solid bottom ring")
        self._connect(self.w_bot_height, "bottom_height")
        f.addRow("Bottom Height", self.w_bot_height)

        g.setLayout(f)
        return g

    def _grp_interface(self):
        """Section 2 - Interface Parameters"""
        g = QGroupBox("2. Interface Parameters (Lamp Socket)")
        f = QFormLayout()

        # Lamp Socket Outer Diameter
        self.w_sock_outer = self._dspin(self.params.socket_outer_diameter, 5, 200, 1, 0.5, "mm",
            "Outer diameter of the lamp socket / neck you're fitting to")
        self._connect(self.w_sock_outer, "socket_outer_diameter")
        f.addRow("Socket Outer Diam", self.w_sock_outer)

        # Socket Wall Thickness
        self.w_sock_wall = self._dspin(self.params.socket_wall_thickness, 0.5, 10, 1, 0.5, "mm",
            "Wall thickness of the socket adapter ring")
        self._connect(self.w_sock_wall, "socket_wall_thickness")
        f.addRow("Socket Wall Thick", self.w_sock_wall)

        # Socket Height
        self.w_sock_height = self._dspin(self.params.socket_height, 1, 200, 1, 1, "mm",
            "Height of the socket adapter cylinder")
        self._connect(self.w_sock_height, "socket_height")
        f.addRow("Socket Height", self.w_sock_height)

        # Tolerance (gap between socket and lamp neck)
        self.w_sock_tol = self._dspin(self.params.socket_tolerance, 0, 2, 2, 0.05, "mm",
            "Clearance gap so the adapter slides onto the lamp neck")
        self._connect(self.w_sock_tol, "socket_tolerance")
        f.addRow("Tolerance", self.w_sock_tol)

        # Lip Height
        self.w_lip_h = self._dspin(self.params.lip_height, 0, 20, 1, 0.5, "mm",
            "Height of the retention lip inside the socket")
        self._connect(self.w_lip_h, "lip_height")
        f.addRow("Lip Height", self.w_lip_h)

        # Lip Width
        self.w_lip_w = self._dspin(self.params.lip_width, 0, 10, 1, 0.5, "mm",
            "Width (overhang) of the retention lip")
        self._connect(self.w_lip_w, "lip_width")
        f.addRow("Lip Width", self.w_lip_w)

        g.setLayout(f)
        return g

    def _grp_spokes(self):
        """Section 3 - Spoke Parameters"""
        g = QGroupBox("3. Spoke Parameters")
        f = QFormLayout()

        # Number of Spokes
        self.w_spoke_count = self._ispin(self.params.spoke_count, 0, 16,
            "Number of spokes connecting socket to shade (0 = no spokes)")
        self._connect(self.w_spoke_count, "spoke_count")
        f.addRow("Number of Spokes", self.w_spoke_count)

        # Spoke Width
        self.w_spoke_width = self._dspin(self.params.spoke_width, 1, 50, 1, 0.5, "mm",
            "Width of each spoke")
        self._connect(self.w_spoke_width, "spoke_width")
        f.addRow("Spoke Width", self.w_spoke_width)

        # Spoke Thickness
        self.w_spoke_thick = self._dspin(self.params.spoke_thickness, 0.5, 20, 1, 0.5, "mm",
            "Thickness (height) of each spoke")
        self._connect(self.w_spoke_thick, "spoke_thickness")
        f.addRow("Spoke Thickness", self.w_spoke_thick)

        g.setLayout(f)
        return g

    def _grp_images(self):
        g = QGroupBox("Images")
        self._img_grid = QGridLayout()
        self._img_grid.setSpacing(6)
        self._rebuild_image_slots()
        g.setLayout(self._img_grid)
        self.images_group = g
        return g

    def _rebuild_image_slots(self):
        # clear grid
        while self._img_grid.count():
            item = self._img_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.img_labels     = []
        self.preview_labels = []
        for i in range(self.panel_count):
            col = QWidget()
            vl  = QVBoxLayout(col)
            vl.setContentsMargins(0,0,0,0)
            vl.setSpacing(4)
            btn = QPushButton(f"Load Photo {i+1}")
            btn.clicked.connect(lambda _, idx=i: self.select_image(idx))
            lbl = QLabel("No file")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            thumb = QLabel("Preview")
            thumb.setFixedSize(100, 100)
            thumb.setAlignment(Qt.AlignCenter)
            vl.addWidget(btn)
            vl.addWidget(lbl)
            vl.addWidget(thumb)
            self._img_grid.addWidget(col, 0, i)
            self.img_labels.append(lbl)
            self.preview_labels.append(thumb)

    def _grp_actions(self):
        g   = QGroupBox("Export")
        row = QHBoxLayout()
        b1  = QPushButton("Generate")
        b1.clicked.connect(self.generate_model)
        b2  = QPushButton("Save STL (shade + socket)…")
        b2.setToolTip("Saves _shade.stl and _socket.stl as separate files")
        b2.clicked.connect(self.export_all_stl)
        b3  = QPushButton("Save Shade only…")
        b3.clicked.connect(self.export_shade_stl)
        row.addWidget(b1); row.addWidget(b2); row.addWidget(b3)
        row.addStretch(1)
        g.setLayout(row)
        return g

    def _grp_view(self):
        row = QHBoxLayout()
        for name in ["Iso","Front","Back","Left","Right","Top","Bottom"]:
            b = QPushButton(name)
            b.setFixedWidth(58)
            b.clicked.connect(lambda _, n=name: self.gl_widget.set_view(n))
            row.addWidget(b)
        row.addStretch(1)
        w = QWidget(); w.setLayout(row)
        return w

    # ------------------------------------------------------------------ UI build
    def _build_ui(self):
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.addWidget(self._grp_images())
        lv.addWidget(self._grp_lithophane())
        lv.addWidget(self._grp_interface())
        lv.addWidget(self._grp_spokes())
        lv.addWidget(self._grp_actions())
        lv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(340)

        self.gl_widget = GLWidget()

        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.addWidget(self._grp_view())
        rv.addWidget(self.gl_widget, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        cw = QWidget()
        QHBoxLayout(cw).addWidget(splitter)
        self.setCentralWidget(cw)

    # ------------------------------------------------------------------ slots
    def _on_sides_changed(self, n):
        self.panel_count = n
        self.params.num_sides = n
        self.image_paths = (self.image_paths + [None]*n)[:n]
        self._rebuild_image_slots()
        for i, p in enumerate(self.image_paths):
            if p:
                name = p.split("/")[-1]
                self.img_labels[i].setText(name[:28])
                pix = load_thumbnail_pixmap(p, 100, 100)
                if pix: self.preview_labels[i].setPixmap(pix)
        self._regen_timer.start()

    def select_image(self, idx):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.image_paths[idx] = path
        name = path.split("/")[-1]
        self.img_labels[idx].setText(name[:28] + ("..." if len(name) > 28 else ""))
        pix = load_thumbnail_pixmap(path, 100, 100)
        if pix: self.preview_labels[idx].setPixmap(pix)
        else:   self.preview_labels[idx].setText("Preview")
        self.generate_model()

    def generate_model(self):
        imgs = [
            Image.open(p).convert("L") if p else None
            for p in self.image_paths
        ]
        self.params.num_sides = self.panel_count

        # high-res build
        bhi = LithophaneBuilder(self.params)
        V, I, N, C = bhi.build(imgs)
        self._shade_mesh  = (V, I, N, C)
        self._socket_mesh = bhi.build_socket()

        highres = _geo_dict(V, I, N, C)

        # low-res preview (4x coarser)
        lo_p = dataclasses.replace(
            self.params, resolution_mm=self.params.resolution_mm * 4)
        blo = LithophaneBuilder(lo_p)
        Vl, Il, Nl, Cl = blo.build(imgs)
        lowres = _geo_dict(Vl, Il, Nl, Cl)

        self.gl_widget.set_geometries(highres, lowres)

    def export_all_stl(self):
        if self._shade_mesh is None:
            self.generate_model()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save STL", "lithophane.stl", "STL Binary (*.stl)")
        if not path:
            return
        try:
            written = save_all_stl(path, self._shade_mesh, self._socket_mesh)
            QMessageBox.information(self, "Saved",
                "Files:\n" + "\n".join(written.values()))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def export_shade_stl(self):
        if self._shade_mesh is None:
            self.generate_model()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Shade STL", "shade.stl", "STL Binary (*.stl)")
        if not path:
            return
        try:
            V, I = self._shade_mesh[0], self._shade_mesh[1]
            save_binary_stl(path, V, I)
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def _geo_dict(V, I, N, C):
    return dict(
        verts=V.astype(np.float32),
        inds=I.astype(np.uint32),
        norms=N.astype(np.float32),
        cols=C.astype(np.float32),
    )
