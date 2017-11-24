import numpy as np
from PIL import Image
import os


def output_image(colorArray, x, y):
    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape(y, x)).astype(np.uint8), "L") for c in colorArray.components()]
    image_name = input("Image name: ")
    ext = ".png"
    image_path = os.path.join("output", image_name+ext)
    Image.merge("RGB", rgb).save(image_path)
