import numpy as np


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


def unit_vector(v):
    vl = v.length()
    bv = vec3(vl, vl, vl)

    return np.divide(v, bv)
