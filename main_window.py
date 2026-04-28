"""Main window — parameter set matches LithophaneMaker.com Lamp Lithophane."""
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QGridLayout, QDoubleSpinBox, QScrollArea,
    QGroupBox, QFormLayout, QSplitter, QSizePolicy, QMessageBox,
    QSlider, QComboBox, QSpinBox, QCheckBox,
)
from PyQt5.QtCore import Qt
from glwidget import GLWidget
from image_utils import load_thumbnail_pixmap
from builder import LithophaneBuilder, BuildParams
from export_stl import save_all_stl, save_binary_stl
import dataclasses


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lithophane Lamp Maker")
        self.resize(1300, 960)

        self.lampshade_types    = ["Normal", "Sphere", "Flat"]
        self.cur_lampshade_type = "Normal"
        self.panel_count        = 4

        self.image_paths    = [None] * self.panel_count
        self.img_labels     = []
        self.preview_labels = []

        self.params = BuildParams(
            height=163.0, top_diam=170.0, bottom_diam=200.0,
            min_thickness=0.8,  max_thickness=3.0,
            resolution_mm=0.5,
            gamma=2.2,
            contrast=1.0, brightness=0.0,
            num_panels=self.panel_count,
            shade_type=self.cur_lampshade_type,
            gap_mm=2.0, gap_brightness=0.0,
            waves_enabled=False, wave_count=4, wave_height_mm=3.0,
            top_brim_height=8.0,    top_brim_thickness=5.0,
            top_brim_overhang_angle=45.0,
            bottom_brim_height=5.0, bottom_brim_thickness=5.0,
            frame_width=5.0, frame_thickness=3.5,
            socket_enabled=False, socket_inner_diam=32.5,
            socket_wall=3.5, socket_height=25.0,
            socket_lip_height=3.5, socket_lip_overhang=1.5,
            spokes_enabled=False, spoke_count=4,
            spoke_width=6.0, spoke_thickness=6.0,
        )

        self._shade_mesh  = None
        self._socket_mesh = None

        self._build_ui()

    # ------------------------------------------------------------------
    def _form_spin(self, value, lo, hi, decimals=2, step=0.5, suffix="", tip=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setValue(value)
        if suffix:
            w.setSuffix(suffix)
        if tip:
            w.setToolTip(tip)
        w.setMaximumWidth(160)
        w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return w

    def _right_wrap(self, w):
        box = QWidget()
        h   = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addStretch(1)
        h.addWidget(w, 0, Qt.AlignRight | Qt.AlignVCenter)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return box

    # ------------------------------------------------------------------
    def _type_and_panel_group(self):
        box    = QGroupBox("Lampshade Type & Panels")
        layout = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.lampshade_types)
        self.type_combo.setCurrentText(self.cur_lampshade_type)
        self.type_combo.currentTextChanged.connect(self._on_lampshade_type_change)
        self.panel_spin = QSpinBox()
        self.panel_spin.setRange(1, 12)
        self.panel_spin.setValue(self.panel_count)
        self.panel_spin.valueChanged.connect(self._on_panel_count_change)
        layout.addRow("Type:",   self.type_combo)
        layout.addRow("Panels:", self.panel_spin)
        box.setLayout(layout)
        return box

    def _on_lampshade_type_change(self, type_):
        self.cur_lampshade_type = type_
        self.params.shade_type  = type_
        self.generate_model()

    def _on_panel_count_change(self, count):
        self.panel_count       = count
        self.params.num_panels = count
        self.image_paths       = [None] * count
        self._refresh_images_group()
        self.generate_model()

    def _refresh_images_group(self):
        parent = self.images_group.parent()
        idx    = parent.layout().indexOf(self.images_group)
        parent.layout().removeWidget(self.images_group)
        self.images_group.deleteLater()
        self.images_group = self._images_group()
        parent.layout().insertWidget(idx, self.images_group)

    def _images_group(self):
        box  = QGroupBox("Images")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self.img_labels     = []
        self.preview_labels = []
        for i in range(self.panel_count):
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
        self.images_group = box
        return box

    def _geometry_group(self):
        box  = QGroupBox("Lithophane Parameters")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        h    = self._form_spin(self.params.height,        20,  600, 1, 1,    "mm",
                               "Height of the lamp.")
        td   = self._form_spin(self.params.top_diam,      20, 1000, 1, 1,    "mm",
                               "Top diameter of the lampshade.")
        bd   = self._form_spin(self.params.bottom_diam,   20, 1000, 1, 1,    "mm",
                               "Bottom diameter of the lampshade.")
        tmin = self._form_spin(self.params.min_thickness, 0.4,  5,  2, 0.05, "mm",
                               "Minimum wall thickness (brightest areas).")
        tmax = self._form_spin(self.params.max_thickness, 0.8, 10,  2, 0.05, "mm",
                               "Maximum wall thickness (darkest areas).")
        res  = self._form_spin(self.params.resolution_mm, 0.2, 5.0, 2, 0.1,  "mm/px",
                               "Mesh resolution. 0.5 = one vertex per 0.5 mm.\n"
                               "Lower = more detail, larger file, slower generation.")
        gamma = self._form_spin(self.params.gamma, 0.5, 4.0, 1, 0.1, "",
                                "Gamma correction (2.2 = LithophaneMaker default).")

        h.valueChanged.connect(    lambda v: setattr(self.params, "height",        v))
        td.valueChanged.connect(   lambda v: setattr(self.params, "top_diam",      v))
        bd.valueChanged.connect(   lambda v: setattr(self.params, "bottom_diam",   v))
        tmin.valueChanged.connect( lambda v: setattr(self.params, "min_thickness",  v))
        tmax.valueChanged.connect( lambda v: setattr(self.params, "max_thickness",  v))
        res.valueChanged.connect(  lambda v: setattr(self.params, "resolution_mm",  v))
        gamma.valueChanged.connect(lambda v: setattr(self.params, "gamma",          v))

        form.addRow("Height",          self._right_wrap(h))
        form.addRow("Top Diameter",    self._right_wrap(td))
        form.addRow("Bottom Diameter", self._right_wrap(bd))
        form.addRow("Min Thickness",   self._right_wrap(tmin))
        form.addRow("Max Thickness",   self._right_wrap(tmax))
        form.addRow("Resolution",      self._right_wrap(res))
        form.addRow("Gamma",           self._right_wrap(gamma))
        box.setLayout(form)
        return box

    def _image_quality_group(self):
        box  = QGroupBox("Image Quality")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        contrast   = self._form_spin(self.params.contrast,   0.1, 3.0, 2, 0.05, "",
                                     "Contrast multiplier. 1.0 = unchanged.")
        brightness = self._form_spin(self.params.brightness, -0.5, 0.5, 2, 0.02, "",
                                     "Brightness offset. 0 = unchanged.")
        contrast.valueChanged.connect(  lambda v: setattr(self.params, "contrast",   v))
        brightness.valueChanged.connect(lambda v: setattr(self.params, "brightness", v))
        form.addRow("Contrast",   self._right_wrap(contrast))
        form.addRow("Brightness", self._right_wrap(brightness))
        box.setLayout(form)
        return box

    def _gap_group(self):
        box  = QGroupBox("Gap Between Panels")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        gap_mm = self._form_spin(self.params.gap_mm, 0.0, 20.0, 1, 0.5, "mm")
        gap_br = self._form_spin(self.params.gap_brightness, 0.0, 1.0, 2, 0.05, "")
        gap_mm.valueChanged.connect(lambda v: setattr(self.params, "gap_mm",         v))
        gap_br.valueChanged.connect(lambda v: setattr(self.params, "gap_brightness", v))
        form.addRow("Gap Width",      self._right_wrap(gap_mm))
        form.addRow("Gap Brightness", self._right_wrap(gap_br))
        box.setLayout(form)
        return box

    def _waves_group(self):
        box  = QGroupBox("Wave Profile")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.waves_check = QCheckBox("Enable waves")
        self.waves_check.setChecked(self.params.waves_enabled)
        wc = QSpinBox(); wc.setRange(1, 20); wc.setValue(self.params.wave_count)
        wh = self._form_spin(self.params.wave_height_mm, 0.5, 30.0, 1, 0.5, "mm")
        self.waves_check.stateChanged.connect(
            lambda s: (setattr(self.params, "waves_enabled", bool(s)), self.generate_model()))
        wc.valueChanged.connect(lambda v: setattr(self.params, "wave_count",     v))
        wh.valueChanged.connect(lambda v: setattr(self.params, "wave_height_mm", v))
        form.addRow("",            self.waves_check)
        form.addRow("Wave Count",  wc)
        form.addRow("Wave Height", self._right_wrap(wh))
        box.setLayout(form)
        return box

    def _brims_group(self):
        box  = QGroupBox("Brims")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        tbh = self._form_spin(self.params.top_brim_height,         0.0, 100, 2, 0.5, "mm")
        tbt = self._form_spin(self.params.top_brim_thickness,      0.0,  50, 2, 0.5, "mm")
        tbo = self._form_spin(self.params.top_brim_overhang_angle, 0.0,  60, 1, 1.0, "\u00b0")
        bbh = self._form_spin(self.params.bottom_brim_height,      0.0, 100, 2, 0.5, "mm")
        bbt = self._form_spin(self.params.bottom_brim_thickness,   0.0,  50, 2, 0.5, "mm")
        tbh.valueChanged.connect(lambda v: setattr(self.params, "top_brim_height",         v))
        tbt.valueChanged.connect(lambda v: setattr(self.params, "top_brim_thickness",      v))
        tbo.valueChanged.connect(lambda v: setattr(self.params, "top_brim_overhang_angle", v))
        bbh.valueChanged.connect(lambda v: setattr(self.params, "bottom_brim_height",      v))
        bbt.valueChanged.connect(lambda v: setattr(self.params, "bottom_brim_thickness",   v))
        form.addRow("Top Brim Height",    self._right_wrap(tbh))
        form.addRow("Top Brim Thickness", self._right_wrap(tbt))
        form.addRow("Top Overhang Angle", self._right_wrap(tbo))
        form.addRow("Bot Brim Height",    self._right_wrap(bbh))
        form.addRow("Bot Brim Thickness", self._right_wrap(bbt))
        box.setLayout(form)
        return box

    def _frames_group(self):
        box  = QGroupBox("Frames")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        fw = self._form_spin(self.params.frame_width,     0.0, 50, 2, 0.5, "mm")
        ft = self._form_spin(self.params.frame_thickness, 0.0, 50, 2, 0.5, "mm")
        fw.valueChanged.connect(lambda v: setattr(self.params, "frame_width",     v))
        ft.valueChanged.connect(lambda v: setattr(self.params, "frame_thickness", v))
        form.addRow("Frame Width",     self._right_wrap(fw))
        form.addRow("Frame Thickness", self._right_wrap(ft))
        box.setLayout(form)
        return box

    def _socket_group(self):
        box  = QGroupBox("Interface (Lamp Socket Adapter)")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.socket_check = QCheckBox("Enable socket adapter")
        self.socket_check.setChecked(self.params.socket_enabled)
        self.socket_check.setToolTip(
            "Exported as a SEPARATE STL file alongside the shade.")
        sk_id = self._form_spin(self.params.socket_inner_diam,   10, 120, 1, 0.5, "mm",
                                "E27 \u2248 26 mm  |  E14 \u2248 17 mm  |  GU10 \u2248 25 mm")
        sk_w  = self._form_spin(self.params.socket_wall,          0.8, 10, 1, 0.5, "mm")
        sk_h  = self._form_spin(self.params.socket_height,        5,  200, 1, 1,   "mm")
        sk_lh = self._form_spin(self.params.socket_lip_height,    0,   20, 1, 0.5, "mm")
        sk_lo = self._form_spin(self.params.socket_lip_overhang,  0,   20, 1, 0.5, "mm")
        self.socket_check.stateChanged.connect(
            lambda s: (setattr(self.params, "socket_enabled", bool(s)), self.generate_model()))
        sk_id.valueChanged.connect(lambda v: setattr(self.params, "socket_inner_diam",   v))
        sk_w.valueChanged.connect( lambda v: setattr(self.params, "socket_wall",         v))
        sk_h.valueChanged.connect( lambda v: setattr(self.params, "socket_height",       v))
        sk_lh.valueChanged.connect(lambda v: setattr(self.params, "socket_lip_height",   v))
        sk_lo.valueChanged.connect(lambda v: setattr(self.params, "socket_lip_overhang", v))
        form.addRow("",               self.socket_check)
        form.addRow("Inner Bore Diam",self._right_wrap(sk_id))
        form.addRow("Wall Thickness", self._right_wrap(sk_w))
        form.addRow("Adapter Height", self._right_wrap(sk_h))
        form.addRow("Lip Height",     self._right_wrap(sk_lh))
        form.addRow("Lip Overhang",   self._right_wrap(sk_lo))
        self.spokes_check = QCheckBox("Enable spokes")
        self.spokes_check.setChecked(self.params.spokes_enabled)
        sp_n = QSpinBox(); sp_n.setRange(2, 16); sp_n.setValue(self.params.spoke_count)
        sp_w = self._form_spin(self.params.spoke_width, 1.0, 30, 1, 0.5, "mm")
        self.spokes_check.stateChanged.connect(
            lambda s: (setattr(self.params, "spokes_enabled", bool(s)), self.generate_model()))
        sp_n.valueChanged.connect(lambda v: setattr(self.params, "spoke_count", v))
        sp_w.valueChanged.connect(lambda v: setattr(self.params, "spoke_width",  v))
        form.addRow("",            self.spokes_check)
        form.addRow("Spoke Count", sp_n)
        form.addRow("Spoke Width", self._right_wrap(sp_w))
        box.setLayout(form)
        return box

    def _lighting_group(self):
        box    = QGroupBox("Preview Lighting")
        layout = QFormLayout()
        self.light_x          = QSlider(Qt.Horizontal); self.light_x.setRange(-500, 500); self.light_x.setValue(100)
        self.light_y          = QSlider(Qt.Horizontal); self.light_y.setRange(-500, 500); self.light_y.setValue(300)
        self.light_z          = QSlider(Qt.Horizontal); self.light_z.setRange(-500, 500); self.light_z.setValue(200)
        self.intensity_slider = QSlider(Qt.Horizontal); self.intensity_slider.setRange(1, 300); self.intensity_slider.setValue(110)
        self.alpha_slider     = QSlider(Qt.Horizontal); self.alpha_slider.setRange(10, 100); self.alpha_slider.setValue(100)

        def update_light():
            self.gl_widget.set_light_pos([
                self.light_x.value(),
                self.light_y.value(),
                self.light_z.value()])
            self.gl_widget.set_light_intensity(self.intensity_slider.value() / 100.0)

        def update_alpha():
            self.gl_widget.set_alpha(self.alpha_slider.value() / 100.0)

        for sl in [self.light_x, self.light_y, self.light_z,
                   self.intensity_slider, self.alpha_slider]:
            sl.sliderPressed.connect(
                lambda: self.gl_widget.set_active_geometry('lowres'))
            sl.sliderReleased.connect(
                lambda: self.gl_widget.set_active_geometry('highres'))

        self.light_x.valueChanged.connect(update_light)
        self.light_y.valueChanged.connect(update_light)
        self.light_z.valueChanged.connect(update_light)
        self.intensity_slider.valueChanged.connect(update_light)
        self.alpha_slider.valueChanged.connect(update_alpha)

        layout.addRow("Light X",      self.light_x)
        layout.addRow("Light Y",      self.light_y)
        layout.addRow("Light Z",      self.light_z)
        layout.addRow("Intensity",    self.intensity_slider)
        layout.addRow("Transparency", self.alpha_slider)
        box.setLayout(layout)
        return box

    def _actions_group(self):
        box = QGroupBox("Actions")
        row = QHBoxLayout()
        gen = QPushButton("Generate Lampshade")
        gen.setObjectName("primaryButton")
        gen.clicked.connect(self.generate_model)
        row.addWidget(gen)
        exp_all = QPushButton("Save STL (shade + socket)\u2026")
        exp_all.setToolTip("Saves *_shade.stl and *_socket.stl as separate files.")
        exp_all.clicked.connect(self.export_all_stl)
        row.addWidget(exp_all)
        exp_shade = QPushButton("Save Shade only\u2026")
        exp_shade.clicked.connect(self.export_shade_stl)
        row.addWidget(exp_shade)
        row.addStretch(1)
        box.setLayout(row)
        return box

    def _build_ui(self):
        left_col = QWidget()
        left_v   = QVBoxLayout(left_col)
        left_v.addWidget(self._type_and_panel_group())
        left_v.addWidget(self._images_group())
        left_v.addWidget(self._geometry_group())
        left_v.addWidget(self._image_quality_group())
        left_v.addWidget(self._gap_group())
        left_v.addWidget(self._waves_group())
        left_v.addWidget(self._brims_group())
        left_v.addWidget(self._frames_group())
        left_v.addWidget(self._socket_group())
        left_v.addWidget(self._lighting_group())
        left_v.addWidget(self._actions_group())
        left_v.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_col)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMaximumWidth(360)

        self.gl_widget = GLWidget()
        view_bar = QHBoxLayout()
        for name in ["Iso", "Front", "Back", "Left", "Right", "Top", "Bottom"]:
            b = QPushButton(name)
            b.setObjectName("viewButton")
            b.clicked.connect(lambda _, n=name: self.gl_widget.set_view(n))
            view_bar.addWidget(b)
        view_bar.addStretch(1)
        vw = QWidget(); vw.setLayout(view_bar)

        right   = QWidget()
        right_v = QVBoxLayout(right)
        right_v.addWidget(vw)
        right_v.addWidget(self.gl_widget, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        wrapper = QWidget()
        root    = QHBoxLayout(wrapper)
        root.addWidget(splitter)
        self.setCentralWidget(wrapper)

    # ------------------------------------------------------------------
    def select_image(self, idx):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.image_paths[idx] = path
        name = path.split("/")[-1]
        self.img_labels[idx].setText(
            name if len(name) <= 28 else name[:25] + "...")
        pix = load_thumbnail_pixmap(path, 110, 110)
        if pix:
            self.preview_labels[idx].setPixmap(pix)
            self.preview_labels[idx].setToolTip(name)
        else:
            self.preview_labels[idx].setText("Preview")
        self.generate_model()

    def generate_model(self):
        imgs = [
            Image.open(p).convert("L") if p else None
            for p in self.image_paths
        ]
        self.params.shade_type = self.cur_lampshade_type
        self.params.num_panels = self.panel_count

        builder_hi = LithophaneBuilder(self.params)
        V, I, N, C = builder_hi.build(imgs)
        self._shade_mesh = (V, I, N, C)
        highres = dict(
            verts=V.astype(np.float32),
            inds=I.astype(np.uint32),
            norms=N.astype(np.float32),
            cols=C.astype(np.float32),
        )

        lo_params  = dataclasses.replace(
            self.params, resolution_mm=self.params.resolution_mm * 4)
        builder_lo = LithophaneBuilder(lo_params)
        Vlo, Ilo, Nlo, Clo = builder_lo.build(imgs)
        lowres = dict(
            verts=Vlo.astype(np.float32),
            inds=Ilo.astype(np.uint32),
            norms=Nlo.astype(np.float32),
            cols=Clo.astype(np.float32),
        )

        self.gl_widget.set_geometries(highres, lowres)
        self._socket_mesh = builder_hi.build_socket()

    def export_all_stl(self):
        if self._shade_mesh is None:
            self.generate_model()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save STL base name", "lithophane.stl", "STL Binary (*.stl)")
        if not path:
            return
        try:
            written = save_all_stl(path, self._shade_mesh, self._socket_mesh)
            QMessageBox.information(
                self, "Export STL",
                "Files saved:\n" + "\n".join(written.values()))
        except Exception as e:
            QMessageBox.critical(self, "Export STL", f"Failed:\n{e}")

    def export_shade_stl(self):
        if self._shade_mesh is None:
            self.generate_model()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Shade STL", "shade.stl", "STL Binary (*.stl)")
        if not path:
            return
        try:
            V, I = self._shade_mesh[0], self._shade_mesh[1]
            save_binary_stl(path, V, I, header_text="Lithophane Shade (mm)")
            QMessageBox.information(self, "Export STL", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export STL", f"Failed:\n{e}")
