from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
import numpy as np

class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.verts = None
        self.inds  = None
        self.norms = None
        self.cols  = None
        self.last = None
        self.rot_x = 20   # tilt up/down around X
        self.rot_z = 30   # orbit around Y (apply as rotation about Z-axis after X)
        self.zoom  = -200
        self.center= np.zeros(3)

        # Y-up view presets: (rot_x, rot_z)
        self.views = {
            'Iso':   (20, 30),
            'Front': (0,   0),
            'Back':  (0, 180),
            'Left':  (0,  90),
            'Right': (0, -90),
            'Top':   (90,  0),
            'Bottom':(-90, 0),
        }

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2,0.2,0.2,1)
        # Lighting stays configured but will be disabled during draw for unlit grayscale
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0,1000,0,1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.1,1.1,1.1,1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1,1,1,1])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.25,0.25,0.25,1])

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Correct unpacking of center: X= self.center[0], Y= self.center[1]
        glTranslatef(-self.center[0], -self.center[1], self.zoom)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_z, 0, 1, 0)

        if self.verts is not None and len(self.verts) > 0:
            glDisable(GL_LIGHTING)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.verts)

            if self.cols is not None:
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


    def resizeGL(self,w,h):
        glViewport(0,0,w,h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glFrustum(-w/200,w/200,-h/200,h/200,10,10000)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self,e):
        self.last = e.pos()

    def mouseMoveEvent(self,e):
        dx = e.x()-self.last.x(); dy = e.y()-self.last.y()
        self.rot_z += dx*0.5   # horizontal drag rotates around world Y
        self.rot_x += dy*0.5   # vertical drag tilts up/down
        self.last = e.pos()
        self.update()

    def wheelEvent(self,e):
        self.zoom += e.angleDelta().y()/8
        self.zoom = min(max(self.zoom,-1000),-10)
        self.update()

    def update_geometry(self, verts, inds, norms=None, cols=None):
        self.verts = verts
        self.inds  = inds
        self.norms = norms
        self.cols  = cols
        if verts is not None and len(verts):
            minv,maxv = verts.min(0), verts.max(0)
            self.center = (minv+maxv)/2
            d = float(np.linalg.norm(maxv-minv))
            self.zoom = -max(d*1.5, 50.0)
        else:
            self.center = np.zeros(3)
            self.zoom = -200
        self.set_view('Iso')
        self.update()

    def set_view(self, name):
        self.rot_x, self.rot_z = self.views.get(name, (20,30))
        self.update()
