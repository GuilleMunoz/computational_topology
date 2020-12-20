from classes import *
from time import time
import numpy as np
from random import gauss
import json


def points_cloud(t='circ', omega=0.05, num=100):
    """
    Returns a point cloud of a circle or a 8 figure
    Args:
        t (str, default='circ'): 'circ' for the circle

    Returns:
        list
    """
    o = omega
    if t == 'circ':
        f = lambda t: (np.cos(t) + gauss(0, o), np.sin(t) + gauss(0, o))
    else:
        f = lambda t: (np.cos(t) + gauss(0, o), np.cos(t) * np.sin(t) + gauss(0, o))

    ls = np.arange(0,  2 * np.pi, 2 * np.pi/ num)

    xs = np.ones(len(ls))
    ys = np.ones(len(ls))

    for i, x in enumerate(ls):
        xs[i], ys[i] = f(x)
    return list(map(list, zip(xs, ys)))


def saved_SC():
    with open('simplicial') as json_file:
        simplicial_complexes = json.load(json_file)
        return list(simplicial_complexes.keys())


if __name__ == '__main__':
    points = np.array(points_cloud(omega=0.2, num=500))
    alpha = AlphaComplex(points=points)
    start = time()
    alpha.persistent_homology()
    end = time()
    print(end - start)


