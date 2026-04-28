from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import (
    glEnable, glDisable, glClear, glLoadIdentity, glTranslatef,
    glRotatef, glViewport, glMatrixMode, glFrustum, glLoadIdentity,
    glClearColor, glDrawElements,
    glEnableClientState, glDisableClientState,
    glVertexPointer, glNormalPointer, glColorPointer,
    glLightfv, glLightModelfv, glMaterialfv,
    GL_DEPTH_TEST, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_LIGHTING, GL_LIGHT0, GL_POSITION, GL_DIFFUSE, GL_SPECULAR,
    GL_LIGHT_MODEL_AMBIENT, GL_PROJECTION, GL_MODELVIEW,
    GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_COLOR_ARRAY,
    GL_FLOAT, GL_UNSIGNED_INT, GL_TRIANGLES,
    GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glBlendFunc, glDepthMask,
    GL_TRUE, GL_FALSE,
)
import ctypes
import numpy as np


class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        # two geometry slots: highres & lowres
        self._geo = {}          # 'highres' | 'lowres'  ->  dict(verts, inds, norms, cols)
        self._active = 'highres'

        self.rot_x  = 20.0
        self.rot_z  = 30.0
        self.zoom   = -350.0
        self.center = np.zeros(3, dtype=np.float32)
        self.last   = None

        self._light_pos       = [100.0, 300.0, 200.0]
        self._light_intensity = 1.1
        self._alpha           = 1.0

        self.views = {
            'Iso':    (20,  30),
            'Front':  ( 0,   0),
            'Back':   ( 0, 180),
            'Left':   ( 0,  90),
            'Right':  ( 0, -90),
            'Top':    (90,   0),
            'Bottom': (-90,  0),
        }

    # ------------------------------------------------------------------
    # Geometry API  (called from main_window)
    # ------------------------------------------------------------------
    def set_geometries(self, highres: dict, lowres: dict):
        """Accept highres and lowres geometry dicts with keys:
           verts, inds, norms, cols  (all np.ndarray)."""
        self._geo['highres'] = highres
        self._geo['lowres']  = lowres
        self._active = 'highres'
        # auto-zoom to bounding box
        V = highres.get('verts')
        if V is not None and len(V):
            mn, mx = V.min(0), V.max(0)
            self.center = ((mn + mx) / 2.0).astype(np.float32)
            d = float(np.linalg.norm(mx - mn))
            self.zoom = -max(d * 1.5, 50.0)
        self.set_view('Iso')
        self.update()

    def set_active_geometry(self, which: str):
        """Switch between 'highres' and 'lowres' for interactive dragging."""
        if which in self._geo:
            self._active = which
            self.update()

    # keep old single-geometry API working
    def update_geometry(self, verts, inds, norms=None, cols=None):
        geo = dict(verts=verts, inds=inds, norms=norms, cols=cols)
        self.set_geometries(geo, geo)

    # ------------------------------------------------------------------
    # Lighting / display API
    # ------------------------------------------------------------------
    def set_light_pos(self, pos):
        self._light_pos = list(pos)
        self.update()

    def set_light_intensity(self, v: float):
        self._light_intensity = float(v)
        self.update()

    def set_alpha(self, v: float):
        self._alpha = float(np.clip(v, 0.0, 1.0))
        self.update()

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------
    def set_view(self, name: str):
        self.rot_x, self.rot_z = self.views.get(name, (20, 30))
        self.update()

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.18, 0.18, 0.18, 1.0)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.25, 0.25, 0.25, 1.0])
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h
        near, far = 1.0, 20000.0
        top = near * 0.5
        glFrustum(-top * aspect, top * aspect, -top, top, near, far)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(-self.center[0], -self.center[1], self.zoom)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_z, 0, 1, 0)

        # update light
        lp = self._light_pos
        i  = self._light_intensity
        glLightfv(GL_LIGHT0, GL_POSITION, [lp[0], lp[1], lp[2], 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [i, i, i, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [i * 0.5, i * 0.5, i * 0.5, 1.0])

        geo = self._geo.get(self._active)
        if geo is None:
            return
        verts = geo.get('verts')
        inds  = geo.get('inds')
        norms = geo.get('norms')
        cols  = geo.get('cols')
        if verts is None or inds is None or len(verts) == 0 or len(inds) == 0:
            return

        alpha = self._alpha
        use_blend = alpha < 0.999
        if use_blend:
            glEnable(GL_BLEND)
            glDepthMask(GL_FALSE)
        else:
            glDisable(GL_BLEND)
            glDepthMask(GL_TRUE)

        glEnable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0,
                        verts.astype(np.float32).flatten())

        if norms is not None and len(norms):
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, 0,
                            norms.astype(np.float32).flatten())

        if cols is not None and len(cols):
            glDisable(GL_LIGHTING)
            glEnableClientState(GL_COLOR_ARRAY)
            if use_blend:
                # inject alpha into colours
                rgba = np.ones((len(cols), 4), dtype=np.float32)
                rgba[:, :3] = cols.astype(np.float32)
                rgba[:, 3]  = alpha
                glColorPointer(4, GL_FLOAT, 0, rgba.flatten())
            else:
                glColorPointer(3, GL_FLOAT, 0,
                               cols.astype(np.float32).flatten())

        # single draw call with flat index buffer
        flat_inds = inds.astype(np.uint32).flatten()
        glDrawElements(GL_TRIANGLES, len(flat_inds), GL_UNSIGNED_INT,
                       flat_inds)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glEnable(GL_LIGHTING)
        if use_blend:
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

    # ------------------------------------------------------------------
    # Mouse / wheel
    # ------------------------------------------------------------------
    def mousePressEvent(self, e):
        self.last = e.pos()

    def mouseMoveEvent(self, e):
        if self.last is None:
            self.last = e.pos()
            return
        dx = e.x() - self.last.x()
        dy = e.y() - self.last.y()
        self.rot_z += dx * 0.5
        self.rot_x += dy * 0.5
        self.last = e.pos()
        self.update()

    def wheelEvent(self, e):
        self.zoom += e.angleDelta().y() / 8.0
        self.zoom  = float(np.clip(self.zoom, -10000.0, -10.0))
        self.update()

    # expose verts/inds for export_stl compatibility
    @property
    def verts(self):
        geo = self._geo.get('highres')
        return geo['verts'] if geo else None

    @property
    def inds(self):
        geo = self._geo.get('highres')
        return geo['inds'] if geo else None
