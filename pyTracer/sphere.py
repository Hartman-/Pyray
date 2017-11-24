# SPHERE

import hitable
from vector import vec3
import numpy as np


class Sphere(object):
    def __init__(self, center=vec3(0., 0., 0.), radius=0.0):
        self.center = center
        self.radius = radius

    def intersect(self, r, t_min, t_max, rec):
        oc = r.origin() - self.center
        a = np.dot(r.direction(), r.direction())
        b = np.dot(oc, r.direction())
        c = np.dot(oc, oc) - (self.radius*self.radius)
        disc = b*b - a*c

        sq = np.sqrt(np.maximum(0, disc))

        h0 = (-b - sq)/a
        h1 = (-b + sq)/a

        h = np.where((h0 > 0) & (h0 < h1), h0, h1)

        hit = (disc > 0) & (h > 0)
        dist = np.where(hit, h, 1.0e39)

        rec.t = h
        rec.p = r.point_at_parameter(rec.t)
        rec.normal = (rec.p - self.center) / self.radius
        return dist
