# SPHERE

import hitable
from vector import vec3, unit_vector
import numpy as np

import functools as ft


class Sphere(object):
    def __init__(self, center=vec3(0., 0., 0.), radius=0.0):
        self.center = center
        self.radius = radius

    def intersect(self, r, t_min, t_max, rec):
        oc = r.origin() - self.center
        a = r.direction().dot(r.direction())
        b = oc.dot(r.direction())
        c = oc.dot(oc) - (self.radius*self.radius)
        disc = b*b - a*c

        sq = np.sqrt(np.maximum(0, disc))

        h0 = (-b - sq)/a
        h1 = (-b + sq)/a

        h = np.where((h0 > 0) & (h0 < h1), h0, h1)

        hit = (disc > 0) & (h > 0)
        dist = np.where(hit, h, 1.0e39)

        rec.t = dist
        rec.p = r.point_at_parameter(rec.t)
        rec.normal = unit_vector((rec.p - self.center) / vec3(self.radius, self.radius, self.radius))
        return dist
