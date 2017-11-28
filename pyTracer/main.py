import functools as ft
import numpy as np
import random
import time

from vector import vec3, unit_vector
from ray import ray
from sphere import Sphere
from camera import camera

from imageo import output_image

rgb = vec3


# Class to store hit data
class hit_record(object):
    __slots__ = ('t', 'p', 'normal')


def random_in_unit_sphere():
    p = vec3(0.0, 0.0, 0.0)
    while True:
        p = vec3(random.random(), random.random(), random.random())*2.0 - vec3(1.0, 1.0, 1.0)
        if (p.squared_length() >= 1.0):
            break
    return p


def raytrace(r, scene, bounce=0):
    hit_rec = hit_record()

    # Determine the closest hits
    distances = [s.intersect(r, 0.0, 1.0e39, hit_rec) for s in scene]
    nearest = ft.reduce(np.minimum, distances)

    color = rgb(0.05, 0.05, 0.05)

    # unit_direction = unit_vector(r.direction())
    # t = 0.5*(unit_direction.y + 1.0)
    # bg = (vec3(1.0, 1.0, 1.0)*(1.0 - t) + vec3(0.5, 0.7, 1.0)*t)

    for (s, d) in zip(scene, distances):
        p = r.point_at_parameter(d)
        N = (p - s.center) / vec3(s.radius, s.radius, s.radius)
        target = p + N + random_in_unit_sphere()

        # Nc = (vec3(N.x + 1, N.y+1, N.z+1) * 0.5)
        # color += Nc * (nearest != 1.0e39) * (d == nearest)
        # nr = ray(p, target-p)
        lv = np.maximum(N.dot(target), 0)

        # color += vec3(0.05, 0.05, 0.05) + raytrace(nr, scene, bounce=bounce+1)*0.5 * (nearest != 1.0e39) * (d == nearest)
        color += vec3(0.1, 0.1, 0.1) * lv * (nearest != 1.0e39) * (d == nearest)
        # color += raytrace(nr, scene)*0.5 * (nearest != 1.0e39) * (d == nearest)
        print(color.components())
    # color += bg * (distances == nearest)
    # return vec3(nearest, nearest, nearest)
    return color


def main():
    t0 = time.time()

    nx = 400
    ny = 200

    world = [Sphere(vec3(0, 0, -1), 0.5), Sphere(vec3(0, -100.5, -1), 100)]

    # Build array of vectors defined on a normalized plane
    # aspect ratio
    # ratio = float(nx) / float(ny)
    # normalized range
    S = (0., 1., 1., 0.)
    # linearly step through each xy pixel and create vector position
    npx = np.tile(np.linspace(S[0], S[2], nx), ny)
    npy = np.repeat(np.linspace(S[1], S[3], ny), nx)
    npz = np.repeat(0.0, (nx * ny))

    origin = vec3(0.0, 0.0, 0.0)
    color = vec3(0, 0, 0)
    cam = camera()
    # test = ray(origin, lower_left_corner + npx*horizontal + npy*vertical)
    # print(test)

    Q = vec3(npx, npy, npz)
    rdir = Q - origin

    ns = 16
    for s in range(ns):
        u = rdir.x + (random.random() / float(nx))
        v = rdir.y + (random.random() / float(ny))
        r = cam.get_ray(u, v)
        # p = r.point_at_parameter(2.0)
        color += raytrace(r, world)

    color = color / vec3(ns, ns, ns)
    print("Took %s" % (time.time() - t0))

    print(color.components())
    output_image(color, nx, ny)

if __name__ == "__main__":
    main()
