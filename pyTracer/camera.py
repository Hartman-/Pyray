# camera.py

from ray import ray
from vector import vec3
import numpy as np


class camera(object):
    def __init__(self):
        self.llc = vec3(-2.0, -1.0, -1.0)
        self.horz = vec3(4.0, 0.0, 0.0)
        self.vert = vec3(0.0, 2.0, 0.0)
        self.origin = vec3(0.0, 0.0, 0.0)

    def get_ray(self, u, v):
        nu = self.horz * u
        nv = self.vert * v

        l = nu.components()[0].shape
        ox = np.repeat(self.origin.x, l)
        oy = np.repeat(self.origin.y, l)
        oz = np.repeat(self.origin.z, l)
        no = vec3(ox, oy, oz)

        direction = self.llc + nu + nv - no
        return ray(no, direction)
