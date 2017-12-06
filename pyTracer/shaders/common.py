# common.py
from ray import ray
import numpy as np
from vector import vec3, unit_vector


def random_in_unit_sphere(shape):
    p = (vec3(np.random.rand(shape), np.random.rand(shape), np.random.rand(shape))*2.0) - vec3(1.0, 1.0, 1.0)

    while True:
        cond = (p.squared_length() >= 1.0)
        rp = (vec3(np.random.rand(shape), np.random.rand(shape), np.random.rand(shape))*2.0) - vec3(1.0, 1.0, 1.0)
        np.place(p.x, cond, rp.x)
        np.place(p.y, cond, rp.y)
        np.place(p.z, cond, rp.z)
        if not np.any(cond):
            break
    return p


def reflect(v, n):
    return (v - n*(v.dot(n)*2.0))


class material(object):
    def scatter(self, r_in, P, N):
        pass


class lambertian(material):
    def __init__(self, a):
        self.albedo = a

    def scatter(self, r_in, P, N):
        target = P + N + random_in_unit_sphere(P.x.shape[0])
        scattered = ray(P, target-P)
        return scattered


class metal(material):
    def __init__(self, a):
        self.albedo = a

    def scatter(self, r_in, P, N):
        reflected = reflect(unit_vector(r_in.direction()), N)
        scattered = ray(P, reflected)
        # isReflected = (N.dot(scattered.direction()) > 0)
        # return np.where(isReflected, True, False)
        return scattered
