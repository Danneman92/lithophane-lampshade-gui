from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
import numpy as np


class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.verts = self.inds = self.norms = self.cols = None
        self.highres_verts = self.highres_inds = self.highres_norms = self.highres_cols = None
        self.lowres_verts  = self.lowres_inds  = self.lowres_norms  = self.lowres_cols  = None
        self.active_geometry = 'highres'
        self.last            = None
        self.rot_x           = 20
        self.rot_z           = 30
        self.zoom            = -200
        self.center          = np.zeros(3, dtype=np.float32)
        self.views = {
            'Iso':    (20, 30),  'Front':  (0,   0),   'Back':   (0, 180),
            'Left':   (0,  90), 'Right':  (0, -90),
            'Top':    (90,  0), 'Bottom': (-90,  0),
        }
        self.light_pos       = [0, 100, 200]
        self.light_intensity = 1.1
        self.alpha           = 1.0

    def set_alpha(self, val):
        self.alpha = val
        if self.cols is not None and self.cols.shape[1] == 4:
            self.cols[:, 3] = val
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.6, 0.6, 0.6, 1.0])
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(-self.center[0], -self.center[1], self.zoom)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_z, 0, 1, 0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION,
                  [self.light_pos[0], self.light_pos[1], self.light_pos[2], 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [self.light_intensity] * 3 + [1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [min(self.light_intensity, 1.0)] * 3 + [1])
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if self.verts is not None and len(self.verts):
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.verts)
            if self.cols is not None and self.cols.shape[1] == 4:
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(4, GL_FLOAT, 0, self.cols)
            elif self.cols is not None:
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_FLOAT, 0, self.cols)
            if self.norms is not None:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, self.norms)
            for tri in self.inds:
                glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, tri)
            glDisableClientState(GL_VERTEX_ARRAY)
            if self.cols is not None:
                glDisableClientState(GL_COLOR_ARRAY)
            if self.norms is not None:
                glDisableClientState(GL_NORMAL_ARRAY)
        glDisable(GL_BLEND)
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHTING)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-w / 200, w / 200, -h / 200, h / 200, 10, 10000)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, e):
        self.last = e.pos()
        self.set_active_geometry('lowres')

    def mouseReleaseEvent(self, e):
        self.set_active_geometry('highres')

    def mouseMoveEvent(self, e):
        dx = e.x() - self.last.x()
        dy = e.y() - self.last.y()
        self.rot_z += dx * 0.5
        self.rot_x += dy * 0.5
        self.last   = e.pos()
        self.update()

    def wheelEvent(self, e):
        self.zoom += e.angleDelta().y() / 8
        self.zoom  = min(max(self.zoom, -1000), -10)
        self.update()

    def set_geometries(self, highres, lowres):
        self.highres_verts = highres.get('verts')
        self.highres_inds  = highres.get('inds')
        self.highres_norms = highres.get('norms')
        self.highres_cols  = highres.get('cols')
        self.lowres_verts  = lowres.get('verts')
        self.lowres_inds   = lowres.get('inds')
        self.lowres_norms  = lowres.get('norms')
        self.lowres_cols   = lowres.get('cols')
        self.set_active_geometry('highres')
        self._update_view_parameters()

    def set_active_geometry(self, which):
        if which not in ('highres', 'lowres'):
            raise ValueError("which must be 'highres' or 'lowres'")
        self.active_geometry = which
        if which == 'highres':
            self.verts = self.highres_verts
            self.inds  = self.highres_inds
            self.norms = self.highres_norms
            self.cols  = self.highres_cols
        else:
            self.verts = self.lowres_verts
            self.inds  = self.lowres_inds
            self.norms = self.lowres_norms
            self.cols  = self.lowres_cols
        self.update()

    def _update_view_parameters(self):
        if self.highres_verts is not None and len(self.highres_verts):
            minv, maxv = self.highres_verts.min(0), self.highres_verts.max(0)
            self.center = (minv + maxv) / 2.0
            d           = float(np.linalg.norm(maxv - minv))
            self.zoom   = -max(d * 1.5, 50.0)
        else:
            self.center = np.zeros(3, dtype=np.float32)
            self.zoom   = -200
        self.set_view('Iso')
        self.update()

    def update_geometry(self, verts, inds, norms=None, cols=None):
        g = {'verts': verts, 'inds': inds, 'norms': norms, 'cols': cols}
        self.set_geometries(g, g)

    def set_view(self, name):
        self.rot_x, self.rot_z = self.views.get(name, (20, 30))
        self.update()

    def set_light_pos(self, xyz):
        self.light_pos = list(xyz)
        self.update()

    def set_light_intensity(self, val):
        self.light_intensity = val
        self.update()
