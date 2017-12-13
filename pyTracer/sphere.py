# SPHERE

import hitable
from vector import vec3, unit_vector
import numpy as np
from shaders.common import lambertian, metal
import functools as ft


class Sphere(object):
    def __init__(self, center=vec3(0., 0., 0.), radius=0.0, shader=lambertian(0.5)):
        self.center = center
        self.radius = radius
        self.shader = shader

    def intersect(self, r, t_min, t_max):
        oc = r.origin() - self.center
        a = r.direction().dot(r.direction())
        b = oc.dot(r.direction())
        c = oc.dot(oc) - (self.radius*self.radius)
        disc = b*b - a*c

        sq = np.sqrt(np.maximum(0, disc))

        h0 = (-b - sq)/a
        h1 = (-b + sq)/a

        h = np.where((h0 > t_min) & (h0 < h1), h0, h1)

        hit = (disc > t_min) & (h > t_min)
        dist = np.where(hit, h, t_max)
        return dist

    def material(self):
        return self.shader

    def light(self, scene):
        return self.center
