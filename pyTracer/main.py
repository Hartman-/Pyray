import functools as ft
import numpy as np
import random
import sys
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


def raytrace(r, scene):

    # Determine the closest hits
    distances = [s.intersect(r, 0.001, 1.0e39) for s in scene]
    nearest = ft.reduce(np.minimum, distances)

    # Ambient
    color = rgb(0.0, 0.0, 0.0)

    unit_dir = unit_vector(r.direction())
    t = (unit_dir.y + 1.0)*0.5
    bgc = (vec3(1.0, 1.0, 1.0)*(1.0 - t) + vec3(0.5, 0.7, 1.0)*t)
    bgc = vec3(0.5, 0.7, 1.0) * t
    for (s, d) in zip(scene, distances):
        hit = (nearest != 1.0e39) & (d == nearest)
        # print(s.radius)
        # print(hit.shape)
        if np.any(hit):
            dc = np.extract(hit, d)
            oc = r.origin().extract(hit)
            dirc = r.direction().extract(hit)
            er = ray(oc, dirc)

            p = r.point_at_parameter(d)
            pe = er.point_at_parameter(dc)

            N = (pe - s.center) / vec3(s.radius, s.radius, s.radius)

            randunit = random_in_unit_sphere(dc.shape[0])
            target = pe + N
            # print(target.components())
            # output_image(target-pe, 400, 200)
            # tc = (vec3(target.x + 1, target.y+1, target.z+1) * 0.5)
            # color += Nc * (nearest != 1.0e39) * (d == nearest)
            nr = ray(pe, target-pe)
            cc = raytrace(nr, scene)*0.5

            color += cc.place2(hit, bgc)
            # color += tc * hit.astype(float)
    return color


def main():
    t0 = time.time()

    nx = 600
    ny = 300

    world = [Sphere(vec3(0, 0, -1), 0.5), Sphere(vec3(0, -100.5, -1), 100)]
    # world = [Sphere(vec3(0, -100.5, -1), 100)]
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
    # print(test)

    Q = vec3(npx, npy, npz)
    rdir = Q - origin

    ns = 32
    for s in range(ns):
        u = rdir.x + (random.random() / float(nx))
        v = rdir.y + (random.random() / float(ny))
        r = cam.get_ray(u, v)
        # p = r.point_at_parameter(2.0)
        color += raytrace(r, world)

    color = color / vec3(ns, ns, ns)
    # color = vec3(np.sqrt(color.x), np.sqrt(color.y), np.sqrt(color.z)) 
    print("Took %s" % (time.time() - t0))

    output_image(color, nx, ny)

if __name__ == "__main__":
    main()
