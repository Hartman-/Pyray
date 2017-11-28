# common.py

import random

from vector import vec3


def random_in_unit_sphere():
    p = vec3(0.0, 0.0, 0.0)
    while (p.squared_length() >= 1.0):
        p = vec3(random.random(), random.random(), random.random())*2.0 - vec3(1.0, 1.0, 1.0)

    return p
