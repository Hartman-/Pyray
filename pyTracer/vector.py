import numpy as np
import copy


class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other):
        return vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self):
        return self.dot(self)

    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def length(self):
        return np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def components(self):
        return (self.x, self.y, self.z)

    def squared_length(self):
        return self.x*self.x + self.y*self.y + self.z*self.z

    def extract(self, cond):
        return vec3(np.extract(cond, self.x),
                    np.extract(cond, self.y),
                    np.extract(cond, self.z))

    def place(self, cond):
        z = np.zeros(cond.shape)
        r = vec3(z, z, z)
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r

    def place2(self, cond, bg):
        r = vec3(bg.x, bg.y, bg.z)
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


def unit_vector(v):
    vl = v.length()
    bv = vec3(vl, vl, vl)

    return np.divide(v, bv)
