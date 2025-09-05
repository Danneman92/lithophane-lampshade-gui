import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QGridLayout, QDoubleSpinBox, QScrollArea, QGroupBox, QFormLayout,
    QSplitter, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt
from glwidget import GLWidget
from image_utils import load_thumbnail_pixmap
from builder import LithophaneBuilder, BuildParams
from export_stl import save_binary_stl


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lithophane Lampshade Creator")
        self.resize(1280, 900)

        # state
        self.image_paths = [None]*4
        self.img_labels = []
        self.preview_labels = []

        # defaults
        self.params = BuildParams(
            height=150.0,
            top_diam=160.0,
            bottom_diam=160.0,
            min_thickness=0.30,
            max_thickness=3.00,
            top_brim_height=6.0,
            top_brim_thickness=2.0,
            bottom_brim_height=6.0,
            bottom_brim_thickness=2.0,
            frame_width=3.0,
            frame_thickness=2.0,
            nrows=120,
            ncols=160,
            num_panels=4
        )

        self._build_ui()

    def _form_spin(self, value, lo, hi, decimals=2, step=0.5, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setValue(value)
        if suffix:
            w.setSuffix(f" {suffix}")
        w.setMaximumWidth(160)
        w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return w

    def _right_wrap(self, w: QWidget) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addStretch(1)
        h.addWidget(w, 0, Qt.AlignRight | Qt.AlignVCenter)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return box

    def _images_group(self):
        box = QGroupBox("Images")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        for i in range(4):
            left_stack = QWidget()
            ls = QVBoxLayout(left_stack)
            ls.setContentsMargins(0, 0, 0, 0)
            ls.setSpacing(4)

            load_btn = QPushButton(f"Load Photo {i+1}")
            load_btn.setObjectName("loadButton")
            load_btn.clicked.connect(lambda _, idx=i: self.select_image(idx))

            name = QLabel("No file")
            name.setObjectName("fileLabel")
            name.setAlignment(Qt.AlignCenter)
            name.setWordWrap(True)
            name.setMinimumWidth(120)
            name.setMaximumWidth(200)

            ls.addWidget(load_btn)
            ls.addWidget(name)

            thumb = QLabel()
            thumb.setFixedSize(110, 110)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setObjectName("thumbnail")
            thumb.setText("Preview")
            thumb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            grid.addWidget(left_stack, i, 0, alignment=Qt.AlignTop)
            grid.addWidget(thumb,      i, 1, alignment=Qt.AlignTop)

            self.img_labels.append(name)
            self.preview_labels.append(thumb)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        box.setLayout(grid)
        return box

    def _geometry_group(self):
        box = QGroupBox("Geometry")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        h = self._form_spin(self.params.height, 20, 600, 1, 1, "mm")
        h.valueChanged.connect(lambda v: setattr(self.params, "height", v))
        td = self._form_spin(self.params.top_diam, 20, 1000, 1, 1, "mm")
        td.valueChanged.connect(lambda v: setattr(self.params, "top_diam", v))
        bd = self._form_spin(self.params.bottom_diam, 20, 1000, 1, 1, "mm")
        bd.valueChanged.connect(lambda v: setattr(self.params, "bottom_diam", v))

        tmin = self._form_spin(self.params.min_thickness, 0.1, 20.0, 2, 0.05, "mm")
        tmin.valueChanged.connect(lambda v: setattr(self.params, "min_thickness", v))
        tmax = self._form_spin(self.params.max_thickness, 0.1, 20.0, 2, 0.05, "mm")
        tmax.valueChanged.connect(lambda v: setattr(self.params, "max_thickness", v))

        form.addRow("Height", self._right_wrap(h))
        form.addRow("Top Diameter", self._right_wrap(td))
        form.addRow("Bottom Diameter", self._right_wrap(bd))
        form.addRow("Min Thickness", self._right_wrap(tmin))
        form.addRow("Max Thickness", self._right_wrap(tmax))
        box.setLayout(form)
        return box

    def _brims_group(self):
        box = QGroupBox("Brims")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        tbh = self._form_spin(self.params.top_brim_height, 0.0, 100.0, 2, 0.5, "mm")
        tbh.valueChanged.connect(lambda v: setattr(self.params, "top_brim_height", v))
        tbt = self._form_spin(self.params.top_brim_thickness, 0.0, 50.0, 2, 0.5, "mm")
        tbt.valueChanged.connect(lambda v: setattr(self.params, "top_brim_thickness", v))

        bbh = self._form_spin(self.params.bottom_brim_height, 0.0, 100.0, 2, 0.5, "mm")
        bbh.valueChanged.connect(lambda v: setattr(self.params, "bottom_brim_height", v))
        bbt = self._form_spin(self.params.bottom_brim_thickness, 0.0, 50.0, 2, 0.5, "mm")
        bbt.valueChanged.connect(lambda v: setattr(self.params, "bottom_brim_thickness", v))

        form.addRow("Top Brim Height", self._right_wrap(tbh))
        form.addRow("Top Brim Thickness", self._right_wrap(tbt))
        form.addRow("Bottom Brim Height", self._right_wrap(bbh))
        form.addRow("Bottom Brim Thickness", self._right_wrap(bbt))
        box.setLayout(form)
        return box

    def _frames_group(self):
        box = QGroupBox("Frames")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        fw = self._form_spin(self.params.frame_width, 0.0, 50.0, 2, 0.5, "mm")
        fw.valueChanged.connect(lambda v: setattr(self.params, "frame_width", v))
        ft = self._form_spin(self.params.frame_thickness, 0.0, 50.0, 2, 0.5, "mm")
        ft.valueChanged.connect(lambda v: setattr(self.params, "frame_thickness", v))

        form.addRow("Frame Width", self._right_wrap(fw))
        form.addRow("Frame Thickness", self._right_wrap(ft))
        box.setLayout(form)
        return box

    def _actions_group(self):
        box = QGroupBox("Actions")
        row = QHBoxLayout()

        gen = QPushButton("Generate Lampshade")
        gen.setObjectName("primaryButton")
        gen.clicked.connect(self.generate_model)
        row.addWidget(gen)

        export = QPushButton("Save as STL…")
        export.clicked.connect(self.export_stl)
        row.addWidget(export)

        row.addStretch(1)
        box.setLayout(row)
        return box

    def _build_ui(self):
        left_col = QWidget()
        left_v = QVBoxLayout(left_col)
        left_v.addWidget(self._images_group())
        left_v.addWidget(self._geometry_group())
        left_v.addWidget(self._brims_group())
        left_v.addWidget(self._frames_group())
        left_v.addWidget(self._actions_group())
        left_v.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_col)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMaximumWidth(320)

        self.gl_widget = GLWidget()
        view_bar = QHBoxLayout()
        for name in ["Iso", "Front", "Back", "Left", "Right", "Top", "Bottom"]:
            b = QPushButton(name)
            b.setObjectName("viewButton")
            b.clicked.connect(lambda _, n=name: self.gl_widget.set_view(n))
            view_bar.addWidget(b)
        view_bar.addStretch(1)

        right = QWidget()
        right_v = QVBoxLayout(right)
        vw = QWidget(); vw.setLayout(view_bar)
        right_v.addWidget(vw)
        right_v.addWidget(self.gl_widget, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        wrapper = QWidget()
        root = QHBoxLayout(wrapper)
        root.addWidget(splitter)
        self.setCentralWidget(wrapper)

    def select_image(self, idx: int):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        self.image_paths[idx] = path
        name = path.split("/")[-1]
        self.img_labels[idx].setText(name if len(name) <= 28 else name[:25] + "...")
        pix = load_thumbnail_pixmap(path, 110, 110)
        if pix:
            self.preview_labels[idx].setPixmap(pix)
            self.preview_labels[idx].setToolTip(name)
        else:
            self.preview_labels[idx].setText("Preview")

    def generate_model(self):
        imgs = [Image.open(p).convert("L") if p else None for p in self.image_paths]
        builder = LithophaneBuilder(self.params)
        V, I, N, C = builder.build(imgs)
        self.gl_widget.update_geometry(
            V.astype(np.float32), I.astype(np.uint32),
            N.astype(np.float32), C.astype(np.float32)
        )

    def export_stl(self):
        # Ensure geometry exists: check attribute presence and nonzero size safely
        has_verts = hasattr(self.gl_widget, "verts") and isinstance(self.gl_widget.verts, (list, tuple, np.ndarray))
        has_inds  = hasattr(self.gl_widget, "inds")  and isinstance(self.gl_widget.inds,  (list, tuple, np.ndarray))

        verts_ok = has_verts and (np.size(self.gl_widget.verts) > 0)
        inds_ok  = has_inds  and (np.size(self.gl_widget.inds)  > 0)

        if not (verts_ok and inds_ok):
            self.generate_model()
            verts_ok = hasattr(self.gl_widget, "verts") and (np.size(self.gl_widget.verts) > 0)
            inds_ok  = hasattr(self.gl_widget, "inds")  and (np.size(self.gl_widget.inds)  > 0)
            if not (verts_ok and inds_ok):
                QMessageBox.warning(self, "Export STL", "No geometry to export.")
                return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save as STL", "lithophane.stl", "STL Binary (*.stl)"
        )
        if not path:
            return

        try:
            save_binary_stl(
                path,
                self.gl_widget.verts,   # will be coerced to (N,3)
                self.gl_widget.inds,    # will be coerced to (M,3)
                header_text="Lithophane (mm)",
                smooth_inside=True
            )
            QMessageBox.information(self, "Export STL", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export STL", f"Failed to save STL:\n{e}")
