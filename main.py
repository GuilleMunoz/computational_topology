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
    trian = [[0, 1, 4], [1, 2, 5], [0, 2, 3], [0, 3, 4], [1, 4, 5], [2, 5, 3],
              [3, 4, 7], [4, 5, 8], [5, 3, 1], [3, 1, 7], [4, 7, 8], [5, 8, 1],
              [1, 1, 7], [2, 7, 8], [1, 8, 0], [1, 2, 7], [0, 2, 8],
              [9, 10, 11], [1, 10, 12], [1, 9, 11], [10, 11, 12], [1, 1, 12],
              [1, 11, 14], [11, 12, 15], [1, 12, 13], [11, 13, 14], [11, 14, 15],
              [12, 13, 15], [10, 13, 14], [1, 14, 15], [9, 15, 13], [9, 10, 13],
              [10, 1, 14], [1, 9, 15]]

    sc = SimplicialComplex(simplices=trian)
    sc.dump("genus 2")
