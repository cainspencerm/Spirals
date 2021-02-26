import numpy
from numba import cuda
from PIL import Image
from numpy.random import randint
import math
from os import getcwd, sep

SCALE = 1
WIDTH = 3840 * SCALE
HEIGHT = 2160 * SCALE


@cuda.jit
def create_image_cuda(pixels, centerX, centerY, symmetry, curve, curve_multiplier, maxRadius):
    i, j = cuda.grid(2)
    if i < pixels.shape[0] and j < pixels.shape[1]:
        x = j - centerX
        y = -(i - centerY)

        r = math.sqrt(float(x ** 2 + y ** 2))
        if r > maxRadius:
            return

        theta = 180 * math.atan2(float(x), float(y)) / numpy.pi
        thetaShift = (curve * r / maxRadius) * multiplier(r, curve_multiplier)
        theta += thetaShift
        theta = theta % 360

        paint(pixels, i, j, theta, symmetry)


@cuda.jit(device=True)
def paint(pixels, i, j, theta, symmetry):
    maxTheta = 360 / symmetry
    pixels[i][j] = (theta % maxTheta) * 256 / maxTheta

    # Invert colors
    if math.sin(theta * symmetry / 100) < .5:
        pixels[i][j] = 255 - pixels[i][j][0]


@cuda.jit(device=True)
def multiplier(radius, i):
    if i == 0:
        return math.log(radius)
    elif i == 1:
        return math.sin(radius / 100)
    elif i == 2:
        return math.cos(radius / 100)
    else:
        return math.tan(radius / 100)


def main():
    choice = input("[N]ew or [I]mport: ")
    if choice == 'N':
        CENTER_X = numpy.random.randint(WIDTH // 3, 2 * WIDTH // 3)
        CENTER_Y = numpy.random.randint(HEIGHT // 3, 2 * HEIGHT // 3)
        SYMMETRY = randint(1, 10)
        CURVE = randint(-2 * numpy.pi * 1000, 2 * numpy.pi * 1000) / 100
        CURVE_MULTIPLIER = randint(0, 2)
        MAX_RADIUS = numpy.random.randint(1500 * SCALE, 2000 * SCALE)

        print("Center: ", (CENTER_X, CENTER_Y))
        print("Symmetry: ", SYMMETRY)
        print("Curve: ", CURVE)
        print("Curve multiplier: ", CURVE_MULTIPLIER)
        print("Max radius: ", MAX_RADIUS)

    else:
        CENTER_X = int(input("Center X: "))
        CENTER_Y = int(input("Center Y: "))
        SYMMETRY = int(input("Symmetry: "))
        CURVE = float(input("Curve: "))
        CURVE_MULTIPLIER = int(input("Curve Multiplier: "))
        MAX_RADIUS = int(input("Max Radius: ")) * SCALE

    pixels = numpy.zeros((HEIGHT, WIDTH, 3), dtype=numpy.uint8)

    threadsperblock = (16, 16)
    blockspergrid_x = int(numpy.ceil(pixels.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(numpy.ceil(pixels.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    create_image_cuda[blockspergrid, threadsperblock](
        pixels,
        CENTER_X,
        CENTER_Y,
        SYMMETRY,
        CURVE,
        CURVE_MULTIPLIER,
        MAX_RADIUS
    )

    image = Image.fromarray(pixels)

    root = getcwd()
    path = root + sep + input("Save file as: ")
    image.save(path)


if __name__ == '__main__':
    main()
