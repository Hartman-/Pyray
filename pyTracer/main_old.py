# main_old.py

import numpy as np
import time

from vector import vec3, unit_vector
from ray import ray

from imageo import output_image

rgb = vec3


def basicGradient():
    t0 = time.time()

    nx = 200
    ny = 100
    im = open('output.ppm', 'w+')
    im.write("P3\n{} {}\n255\n".format(nx, ny))

    # Build array of vectors defined on a normalized plane
    # aspect ratio
    r = float(nx) / float(ny)
    # normalized range
    S = (-1., 1. / r + .25, 1., -1. / r + .25)
    # linearly step through each xy pixel and create vector position
    npx = np.tile(np.linspace(S[0], S[2], nx), ny)
    npy = np.repeat(np.linspace(S[1], S[3], ny), nx)

    for y in range(ny):
        for x in range(nx):
            invy = ny - y

            r = float(x) / float(nx)
            g = float(invy) / float(ny)
            b = 0.2

            ir = int(255.99 * r)
            ig = int(255.99 * g)
            ib = int(255.99 * b)
            im.write("{} {} {}\n".format(ir, ig, ib))

    print("Took: %s seconds" % (time.time() - t0))


def makeGradient(C):
    color = C * 255.99
    return color


# This code outputs a world position image... by mistake
def worldPP():
    nx = 400
    ny = 300
    # Build array of vectors defined on a normalized plane
    # aspect ratio
    r = float(nx) / float(ny)
    # normalized range
    S = (-1., 1. / r + .25, 1., -1. / r + .25)
    # linearly step through each xy pixel and create vector position
    npx = np.tile(np.linspace(S[0], S[2], nx), ny)
    npy = np.repeat(np.linspace(S[1], S[3], ny), nx)
    npz = np.repeat(0.2, (nx * ny))

    baseColor = vec3(npx, npy, npz)
    gradientColor = makeGradient(baseColor)

    from PIL import Image
    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape(ny, nx)).astype(np.uint8), "L") for c in gradientColor.components()]
    Image.merge("RGB", rgb).save("output_02.png")


# Simple Hello World graphics gradient implemented using NumPy arrays
# Output achieved using PIL
# Faster (and more scalable) than using simple for loops to loop over pixels
def numpyGradient():
    t0 = time.time()

    nx = 400
    ny = 200
    # Build array of vectors defined on a normalized plane
    # aspect ratio
    r = float(nx) / float(ny)
    # normalized range
    S = (0., 1., 1., 0.)
    # linearly step through each xy pixel and create vector position
    npx = np.tile(np.linspace(S[0], S[2], nx), ny)
    npy = np.repeat(np.linspace(S[1], S[3], ny), nx)
    npz = np.repeat(0.2, (nx * ny))

    colorNorm = vec3(npx, npy, npz)
    fullColor = makeGradient(colorNorm)

    print("Took %s" % (time.time() - t0))

    from PIL import Image
    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape(ny, nx)).astype(np.uint8), "L") for c in colorNorm.components()]
    Image.merge("RGB", rgb).save("output_02.png")


def colorOLD(r):
    unit_direction = r.direction().norm()
    t = 0.5*(unit_direction.y + 1.0)
    return vec3(1.0, 1.0, 1.0)*(1.0 - t) + vec3(0.5, 0.7, 1.0)*t    


def simpleRaycast():
    t0 = time.time()

    nx = 200
    ny = 100
    # Build array of vectors defined on a normalized plane
    # aspect ratio
    r = float(nx) / float(ny)
    # normalized range
    S = (0., 1., 1., 0.)
    # linearly step through each xy pixel and create vector position
    npx = np.tile(np.linspace(S[0], S[2], nx), ny)
    npy = np.repeat(np.linspace(S[1], S[3], ny), nx)
    npz = np.repeat(0.0, (nx * ny))

    origin = vec3(0.0, 0.0, 0.0)

    # test = ray(origin, lower_left_corner + npx*horizontal + npy*vertical)
    # print(test)

    Q = vec3(npx, npy, npz)
    rdir = Q - origin

    lower_left_corner = vec3(-2.0, -1.0, -1.0)
    horizontal = vec3(4.0, 0.0, 0.0)
    vertical = vec3(0.0, 2.0, 0.0)

    u = horizontal * rdir.x
    v = vertical * rdir.y
    direction = lower_left_corner + u + v
    iray = ray(origin, direction)
    colorRet = color(iray)

    print("Took %s" % (time.time() - t0))

    from PIL import Image
    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape(ny, nx)).astype(np.uint8), "L") for c in colorRet.components()]
    Image.merge("RGB", rgb).save("output/output_simpleRayCast.png")


def hit_sphere_bad(center, radius, r):
    oc = r.origin() - center
    a = r.direction().dot(r.direction())
    b = 2.0 * oc.dot(r.direction())
    c = oc.dot(oc) - radius*radius
    disc = b*b - a*c*4.0
    return np.where((disc < 0) > 0, -1, 1)


def hit_sphere(center, radius, r):
    oc = r.origin() - center
    a = r.direction().dot(r.direction())
    b = 2.0 * oc.dot(r.direction())
    c = oc.dot(oc) - radius*radius
    disc = b*b - a*c*4.0

    sq = np.sqrt(np.maximum(0, disc))
    # h0 = (-b - sq) / 2
    # h1 = (-b + sq) / 2
    # h = np.where((h0 > 0) & (h0 < h1), h0, h1)

    # hit = (disc > 0) & (h > 0)
    return np.where(disc < 0, -1.0, (-b - sq) / (a * 2.0))

def color(r):
    hitpts = hit_sphere(vec3(0.0, 0.0, -1.0), 0.4, r)

    unit_direction = r.direction().norm()
    t = 0.5*(unit_direction.y + 1.0)
    
    color = (vec3(1.0, 1.0, 1.0)*(1.0 - t) + vec3(0.5, 0.7, 1.0)*t)*hitpts + (vec3(1.0, 0.0, 0.0)*(1 - hitpts))
    return(color)


def raytrace_OLD(r, scene):
    hitpts = hit_sphere(vec3(0.0, 0.0, -1.0), 0.4, r)
    m = np.copy(hitpts)

    unit_direction = unit_vector(r.direction())
    t = 0.5*(unit_direction.y + 1.0)

    mask = np.where(m > 0.0, 0, 1)

    N = (r.point_at_parameter(t) - vec3(0.0, 0.0, -1.0)) * (1.0 / 0.4)
    Nc = (vec3(N.x + 1, N.y+1, N.z+1) * 0.5)
    color = (vec3(1.0, 1.0, 1.0)*(1.0 - t) + vec3(0.5, 0.7, 1.0)*t)*mask + (Nc * (1-mask))

    return(color)


def main():
    t0 = time.time()

    nx = 200  
    ny = 100
    # Build array of vectors defined on a normalized plane
    # aspect ratio
    r = float(nx) / float(ny)
    # normalized range
    S = (0., 1., 1., 0.)
    # linearly step through each xy pixel and create vector position
    npx = np.tile(np.linspace(S[0], S[2], nx), ny)
    npy = np.repeat(np.linspace(S[1], S[3], ny), nx)
    npz = np.repeat(0.0, (nx * ny))

    origin = vec3(0.0, 0.0, 0.0)

    # test = ray(origin, lower_left_corner + npx*horizontal + npy*vertical)
    # print(test)

    Q = vec3(npx, npy, npz)
    rdir = Q - origin

    lower_left_corner = vec3(-2.0, -1.0, -1.0)
    horizontal = vec3(4.0, 0.0, 0.0)
    vertical = vec3(0.0, 2.0, 0.0)

    u = horizontal * rdir.x
    v = vertical * rdir.y
    direction = lower_left_corner + u + v
    iray = ray(origin, direction)
    colorRet = color(iray)

    print("Took %s" % (time.time() - t0))

    output_image(colorRet, nx, ny)

if __name__ == "__main__":
    main()
