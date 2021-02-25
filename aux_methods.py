from math import sqrt
import numpy as np


def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def radius(x1, y1, x2, y2, x3, y3):
    """
    Computes de center and radius of 3 points (x1, y1), (x2, y2) and (x3, y3)
    Args:
        x1:
        y1:
        x2:
        y2:
        x3:
        y3:

    Returns (float): circumradius

    """
    A = np.array([[x3 - x1, y3 - y1], [x3 - x2, y3 - y2]])
    Y = np.array([(x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2), (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2)])

    Ainv = np.linalg.inv(A)

    X = 0.5 * np.dot(Ainv, Y)
    x, y = X[0], X[1]

    # Without the max there is an error with the triangle (1, 2, 2, 6, 3, 1.5)
    return max(dist(x1, y1, x2, y2) / 2, dist(x1, y1, x3, y3) / 2, dist(x2, y2, x3, y3) / 2,
               np.sqrt((x - x1) ** 2 + (y - y1) ** 2))


def center_radius(x1, y1, x2, y2):
    """
    Computes de center and radius of two points (x1, y1) and (x2, y2)
    Args:
        x1:
        y1:
        x2:
        y2:

    Returns tuple(tuple(float), float): Center and radius squared

    """
    return ((x1 + x2) / 2, (y1 + y2) / 2), ((x2 - x1) ** 2 + (y2 - y1) ** 2) / 4
