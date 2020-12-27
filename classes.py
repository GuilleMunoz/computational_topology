from itertools import combinations
from functools import cmp_to_key

from math import inf, sqrt

from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation

from jit_method import radius, center_radius, dist

import json
from os.path import getsize as size


class SimplicialComplex:
    """
    A simplicial complex is a set of points, segments, ..., n-dimensional simplices

    Args:
        simplices (list of simplices, default None): List of simplices to inicialice the complex
    """

    def __init__(self, simplices=None):

        self.simplices_maximales = set()

        if not (simplices is None):
            self.extend(simplices)

    def __repr__(self):
        return ',\n'.join(list(map(str, self.simplices_maximales)))

    def add(self, simplex):
        """
        Add simplex to the simplicial complex

        Args:
            simplex (list): set de vertices ([1,2,3])
        """
        simplex_tup = tuple(sorted(simplex))
        simplex = set(simplex)

        if not self.simplices_maximales:  # no simplices yet
            self.simplices_maximales.add(simplex_tup)
            return

        to_del = set()

        for key in self.simplices_maximales:

            key_set = set(key)

            if simplex.issubset(key_set):
                break
            elif key_set.issubset(simplex):
                to_del.add(key)
        else:
            if to_del:
                self.simplices_maximales -= to_del
            self.simplices_maximales.add(simplex_tup)

    def extend(self, simplices):
        """
        Extend the list of simplices

        Args:
            simplices (iterator(list)): list of simplices
            filtration_values (iterator(float)): filtration value of each simplex must have the same length as
                                                 simplices
        """
        for simplex in simplices:
            self.add(simplex)

    def load(self, name, file='simplicial'):
        """
        Loads simplicial from a file as a JSON

        Args:
            name (str): Simplicial name (ej. 'toro')
            file (str, default 'simplicia'): name of the file where the simplicial is stored
        """
        with open(file) as json_file:
            simplicial_complexes = json.load(json_file)
            self.simplices_maximales = set(map(lambda x: tuple(set(x)), simplicial_complexes[name]))

    def dump(self, name, file='simplicial'):
        """
        Dumps simplicial_complex to a file as a JSON

        Args:
            simplicial_complex (SimplicialComplex): The simplicial to dump
            name (str): Simplicial name (ej. 'torus')
            file (str, default 'simplicial'): name of the file where the simplicial complex is stored

        Returns:
            (SimplicialComplex)
        """
        with open(file, 'r+') as json_file:
            if size(file) > 0:
                simplicial_complexes = json.load(json_file)
            else:
                simplicial_complexes = dict()

            simplicial_complexes[name] = list(self.simplices_maximales)
            json_file.seek(0)
            json.dump(simplicial_complexes, json_file)

    def dimension(self):
        """
        Calculate the dimension of the simplicial complex

        Returns:
            (int): dimension
        """
        return max(map(len, self.simplices_maximales)) - 1

    @staticmethod
    def powerset(simplex):
        """
        Returns all the faces of a simplex.
        It does not uses itertools.combinations for all the posible dimentions
        so its much faster than combs.

        Args:
            simplex (iterator(int)): set of points

        Returns:
            list(tuple): Faces of the simplex
        """
        x = len(simplex)
        masks = [1 << i for i in range(x)]
        ls = []
        for i in range(1, 1 << x):
            ls.append(tuple([ss for mask, ss in zip(masks, simplex) if i & mask]))
        return ls

    @staticmethod
    def combs(simplex, n=-1):
        """
        If n >= 0 it returns all n-dimensional combinations of a simplex.
        If n < 0 returns all i-dimensional faces with i <= -n.

        Args:
            simplex (iterator(int)): set of points
            n (int)

        Returns:
            list(tuple): all the faces
        """
        ls = list(simplex)

        if n > len(ls):
            return []

        max_ = n + 1
        if n < 0:
            if n == -1:
                max_ = len(ls)
            else:
                max_ = min(-n, len(ls))

        min_ = max_ if n > -1 else 1
        cs = []

        for size in range(min_, max_ + 1):
            cs.extend(combinations(ls, size))

        return cs

    def faces(self, n=-1):
        """
        If n > 1 returns the n-dimensional simplices of the complex
        If n = -1 returns all the simplices of the complex
        Si n < -1 returns all the i-dimensional simplices with i <= -n ((-n + 1)-skeleton)

        Args:
            n (int)

        Returns:
            set(tuple) : All the faces
        """
        faces = set()

        if n == -1:
            combs = self.powerset
        else:
            combs = lambda s: self.combs(s, n)

        for simplex in map(set, self.simplices_maximales):
            faces.update(combs(simplex))

        return set(faces)

    def skeleton(self, n):
        """
        Returns the n-skeleton

        Args:
            n (int): dimension of the n-skeleton

        Returns:
            (SimplicialComplex): n-skeleton
        """
        return SimplicialComplex(map(list, self.faces(n=n)))

    def star(self, simplex):
        """
        Returns the star of a simplex in the simplicial complex

        Args:
            simplex (set):

        Returns:
            (list(set)): star
        """
        star = {tuple(simplex)}

        for ss in map(set, self.simplices_maximales):

            if not simplex.issubset(ss):
                continue

            ls = list(map(set, self.combs(ss - simplex)))

            star.update(list(map(lambda x: tuple(x | set(simplex)), ls)))

        return list(map(set, star))

    def link(self, simplex):
        """
        Returns the link of a simplex in the simplicial complex

        Args:
            simplex (set):

        Returns:
            (list(set)): link
        """
        link = set()

        for ss in map(set, self.simplices_maximales):

            intersection = ss.intersection(simplex)

            if not intersection:
                continue

            ls = list(map(set, self.combs(ss - intersection)))
            link.update(list(map(tuple, ls)))

        return list(map(set, link))

    def closed_star(self, simplex):
        """
        Returns the closure of a simplex in the simplicial complex

        Args:
            simplex (set):

        Returns:
            (list(set)): star
        """
        estrella_cerrada = set()

        for ss in map(set, self.simplices_maximales):
            for v in simplex:
                if v in ss:
                    estrella_cerrada.update(self.combs(ss))
                    break

        return list(map(set, estrella_cerrada))

    def num_connected_components(self):
        """
        Returns the number of connect components of the simplicial complex.

        Returns:
            (int)
        """
        vertices = {j for i in self.simplices_maximales for j in i}
        vertices = list(map(lambda i: {i}, vertices))

        for simp in self.faces(n=1):

            index = [-1, -1]
            for i, comp in enumerate(vertices):
                if simp[0] in comp:
                    index[0] = i
                if simp[1] in comp:
                    index[1] = i

            if index[0] != index[1]:
                vertices[index[0]] |= vertices[index[1]]
                del vertices[index[1]]

        return len(vertices)

    def euler_characteristic(self):
        """
        Returns the euler characteristic of a simplicial complex.

        Returns:
            (int)
        """
        ls = list(map(lambda ss: 1 if len(ss) % 2 == 1 else -1, self.faces()))
        return sum(ls)

    def edge_matrix_p(self, p):
        """
        Returns the edge (faces of dim p) matrix of the simplicial complex

        Args:
            p (int): dimension of the edges to take

        Returns:
            (np.array): The edge matrix
        """

        faces_p = self.faces(p)

        if p == 0:
            return np.zeros((1, len(faces_p)), dtype=np.bool)

        faces_p_1 = self.faces(p - 1)

        if p == self.dimension() + 1:
            return np.ones((len(faces_p_1), 1), dtype=np.bool)

        def cmp(tup, tup1):
            for i, j in zip(tup, tup1):
                if i > j:
                    return -1
                elif i < j:
                    return 1

        faces_p = list(map(set, sorted(faces_p, key=cmp_to_key(cmp), reverse=True)))
        faces_p_1 = list(map(set, sorted(faces_p_1, key=cmp_to_key(cmp), reverse=True)))

        matrix = np.zeros((len(faces_p_1), len(faces_p)), dtype=np.bool)

        for i in range(len(faces_p_1)):
            for j in range(len(faces_p)):
                if faces_p_1[i].issubset(faces_p[j]):
                    matrix[i, j] = np.True_

        return matrix

    def smith_normal_form(self, p):
        """
        Computes the smith normal form. It also return the rank of Bp-1 (number of 1s) and  rank of Zp

        Args:
            p (int): dimension of the edges to take

        Returns:
            (np.array, int, int): Smith normal form of the edge matrix of p, rank Bp-1, rank Zp
        """

        edge_matrix = self.edge_matrix_p(p)
        dim_Bp1 = 0
        if p != 0:
            for i in range(min(edge_matrix.shape)):
                if not edge_matrix[i, i]:
                    for i_ in range(i, edge_matrix.shape[0]):
                        for j in range(i, edge_matrix.shape[1]):
                            if edge_matrix[i_, j]:
                                edge_matrix[:, [i, j]] = edge_matrix[:, [j, i]]
                                if i != i_:
                                    edge_matrix[[i, i_], :] = edge_matrix[[i_, i], :]
                                break
                        else:
                            continue
                        break
                    else:
                        break

                dim_Bp1 += 1

                for j in range(i + 1, edge_matrix.shape[1]):
                    if edge_matrix[i, j]:
                        edge_matrix[:, j] = edge_matrix[:, i] ^ edge_matrix[:, j]

                for i_ in range(i + 1, edge_matrix.shape[0]):
                    edge_matrix[i_, i] = np.False_

        return edge_matrix, dim_Bp1, edge_matrix.shape[1] - dim_Bp1

    def betti_nums(self):
        """
        Computes the Betti numbers of the simplicial complex, using the Smith normal form
        of the edge matrices.

        Returns:
            (list(int)): list of Betti numbers
        """

        ls_Bp = []
        ls_Zp = []

        for i in range(self.dimension() + 1):
            _, dim_Bp1, dim_Zp = self.smith_normal_form(i)
            ls_Bp.append(dim_Bp1)
            ls_Zp.append(dim_Zp)

        del ls_Bp[0]
        ls_Bp.append(0)

        return [i - j for i, j in zip(ls_Zp, ls_Bp)]

    def betti_nums_incremental_R2(self):
        """
        Computes the Betti numbers of the simplicial complex, using the incremental
        algortithm. (works just in R2)

        Returns:
            (list(int)): list of Betti numbers
        """

        vertices = {j for i in self.simplices_maximales for j in i}
        bettis = [len(vertices), - len(self.faces(n=2)), 0]

        vertices = list(map(lambda i: {i}, vertices))

        for simp in self.faces(n=1):

            index = [-1, -1]
            for i, comp in enumerate(vertices):
                if simp[0] in comp:
                    index[0] = i
                if simp[1] in comp:
                    index[1] = i

            if index[0] != index[1]:
                bettis[0] -= 1
                vertices[index[0]] |= vertices[index[1]]
                del vertices[index[1]]
            else:
                bettis[1] += 1

        return bettis


class Filtration(SimplicialComplex):
    """
    A filtration is a indexed simplicial complex. Each simplex has a given value and the complex is
    ordered by this value

    Args:
        simplices (list of simplices, default None): List of simplices to inicialice the complex
        filtration_values (list(int), default None): filtration value of each simplex must have the
                                                     same length as simplices
    """

    def __init__(self, simplices=None, filtration_values=None):
        super(Filtration, self).__init__()
        self.filtration_values = dict()

        if not (simplices is None):
            self.extend(simplices, filtration_values)

    def __repr__(self):
        filtration = '\n'
        for item in self.filtration_values.items():
            filtration += '{} : {}\n'.format(str(item[0]), item[1])
        max_simpls = '\n\n{}\n{}Maximal simplices\n{}\n\n{}'.format('*' * 68,
                                                                    ' ' * 30,
                                                                    '*' * 68,
                                                                    super(Filtration, self).__repr__())

        filtration = '\n\n{}\n{}Filtration\n{}\n\n{}'.format('*' * 68,
                                                             ' ' * 30,
                                                             '*' * 68,
                                                             filtration)
        return max_simpls + filtration

    def add(self, simplex, filtration=0):
        """
        Add simplex to the simplicial complex

        Args:
            simplex (set): set de vertices ({1,2,3})
            filtration (float, default 0): filtration value it will determine whe it will appear
        """

        super(Filtration, self).add(simplex)
        simplex_tup = tuple(simplex)

        if simplex_tup not in self.filtration_values:
            self.filtration_values[simplex_tup] = filtration
        else:
            self.filtration_values[simplex_tup] = min(filtration, self.filtration_values[simplex_tup])

        for key in self.filtration_values:

            key_set = set(key)

            if simplex.issubset(key_set) and key != simplex_tup:
                self.filtration_values[simplex_tup] = min(self.filtration_values[key],
                                                          self.filtration_values[simplex_tup])

            elif key_set.issubset(simplex):
                self.filtration_values[key] = min(self.filtration_values[key], self.filtration_values[simplex_tup])

        for tup in super(Filtration, self).powerset(simplex):
            if tup in self.filtration_values:
                self.filtration_values[tup] = min(self.filtration_values[tup], self.filtration_values[simplex_tup])
            else:
                self.filtration_values[tup] = self.filtration_values[simplex_tup]

    def extend(self, simplices, filtration_values=None):
        """
        Extend the list of simplices

        simplices (list of simplices, default None): List of simplices to inicialice the complex
        filtration_values (list(int), default None): filtration value of each simplex must have the
                                                     same length as simplices
        """

        if filtration_values is None:
            filtration_values = [0] * len(simplices)

        for simplex, filtration_value in zip(simplices, filtration_values):
            self.add(simplex, filtration_value)

    def set_filtration_value(self, simplex, filtration_value):
        """
        Set the filtration value of a given simplex

        Args:
            simplex:
            filtration_value:
        """
        if tuple(simplex) in self.filtration_values:
            self.filtration_values[tuple(simplex)] = filtration_value

    def dimension(self):
        """
        Calculate the dimension of the simplicial complex

        Returns:
            (int): dimension
        """
        return super(Filtration, self).dimension()

    def faces(self, n=-1):
        """
        If n > 1 returns the n-dimensional simplices of the complex
        If n = -1 returns all the simplices of the complex
        Si n < -1 returns all the i-dimensional simplices with i <= -n ((-n + 1)-skeleton)

        Args:
            n (int, default -1)

        Returns:
            list(tuple) : All the faces
        """

        if n == -1:
            ls = self.filtration_values.keys()
        elif n >= 0:
            ls = [i for i in self.filtration_values.keys() if len(i) == n + 1]
        else:
            ls = [i for i in self.filtration_values.keys() if len(i) <= (-n - 1)]

        return ls

    def euler_caracteristic(self):
        """
        Returns the euler caracteristic of a simplicial complex

        Returns:
            (int)
        """
        return sum(map(lambda ss: 1 if len(ss) % 2 == 1 else -1, self.filtration_values.keys()))

    def get_by_filtration_value(self, filtration_value=inf):
        """
        Return a simplicial complex at an instance (filtration_value). That's the simplices with value less than
        filtration_value.

        Args:
            filtration_value (float, default math.inf): instance

        Returns:
            (Filtration)
        """
        ls = [simp for simp in self.filtration_values.items() if simp[1] <= filtration_value]
        simplices, values = zip(*ls)

        return Filtration(simplices=list(map(set, simplices)), filtration_values=list(values))

    def filtration_order(self):
        """
        Returns the filtration orded by the value

        Returns:
            list(tuple)
        """

        def cmp(x, y):
            if x[1] == y[1]:
                return len(y[0]) - len(x[0])
            return y[1] - x[1]

        return [i[0] for i in sorted(self.filtration_values.items(), key=cmp_to_key(cmp), reverse=True)]

    def edge_matrix(self):
        """
        Returns the generalized edge matrix.

        Returns:
            (np.array) : the edge matrix
            (list(tuple)): the corresponding faces of the edge matrix
        """

        def key(x, y):
            if len(x) == len(y):
                return self.filtration_values[x] - self.filtration_values[y]
            return len(x) - len(y)

        faces = sorted(self.faces(), key=cmp_to_key(key))

        matrix = np.zeros([len(faces), len(faces)], dtype=np.bool)
        faces_ss = list(map(set, faces))
        for j, simp_j in enumerate(faces_ss):
            if len(simp_j) == 1:
                continue
            for i, simp_i in enumerate(faces_ss):
                matrix[i, j] = len(simp_i) == (len(simp_j) - 1) and simp_i.issubset(simp_j)

        return matrix, faces

    def threshold(self, simplex):
        """
        Returns the filtration value of a simplex

        Args:
            simplex (set):
        """

        return self.filtration_values[simplex]

    def threshold_values(self):
        """
            Returns the filtration values of filtration sorted

            Returns:
                (list(float))
        """

        return sorted(set(self.filtration_values.values()))


class VietorisRipsComplex(Filtration):

    def __init__(self, points=None):

        super(VietorisRipsComplex, self).__init__()
        self.points = []

        if points is not None:
            self.add_points(points)

    def add(self, simplex, r=0):
        """
        Add simplex to the simplicial complex

        Args:
            simplex (set): set de vertices ({1,2,3})
            r (float, default 0): filtration value it will determine whe it will appear
        """
        self.filtration_values[tuple(sorted(simplex))] = r

        super(Filtration, self).add(simplex)

    def add_points(self, points):
        """
        Add a list of points in R2

        Args:
            points (iterable(tuples)): iterable of points in R2 (2-tuples) [(x1, y1), (x2, y2), ...]
        """

        for i in range(len(points)):
            self.add((i,))

        for edge in self.combs(range(len(points)), 1):
            self.add(edge, 0.5 * dist(points[edge[0]][0], points[edge[0]][1], points[edge[1]][0], points[edge[1]][1]))

        for i in range(2, len(points)):
            for simp in self.combs(range(len(points)), i):
                max_ = 0
                for edge in self.combs(simp, 1):
                    max_ = max(max_, 0.5 * dist(points[edge[0]][0],
                                               points[edge[0]][1],
                                               points[edge[1]][0],
                                               points[edge[1]][1]))
                self.add(simp, max_)

        self.points = points


class AlphaComplex(Filtration):
    """

    An alpha complex is a filtration constructed from de Delaunay triangulation and
    the radius of the simplices (0 for a 0-simplex, the length for a 1-simplex, and circumdiagonal for a 2-simplex)

    Args:
        points (iterable(tuples)): iterable of points in R2 (2-tuples) [(x1, y1), (x2, y2), ...]

    """

    def __init__(self, points=None):

        super(AlphaComplex, self).__init__()
        self.points = []

        if not points is None:
            self.add_points(points)

    def add(self, simplex, r=0):
        """
        Add simplex to the simplicial complex

        Args:
            simplex (set): set de vertices ({1,2,3})
            r (float, default 0): filtration value it will determine whe it will appear
        """
        self.filtration_values[tuple(sorted(simplex))] = r

        super(Filtration, self).add(simplex)

    def add_points(self, points):
        """
        Add a list of points in R2

        Args:
            points (iterable(tuples)): iterable of points in R2 (2-tuples) [(x1, y1), (x2, y2), ...]
        """
        delaunay = Delaunay(points)

        self.extend(list(map(lambda x: [x], range(len(points)))))

        for simplex in delaunay.simplices:
            x1, y1, x2, y2, x3, y3 = tuple(points[simplex].flatten())
            self.add(simplex, radius(x1, y1, x2, y2, x3, y3))

        ls = np.arange(points.shape[0])
        for (u, v) in super(Filtration, self).faces(n=1):
            c, r2 = center_radius(points[u][0], points[u][1], points[v][0], points[v][1])

            filter_ = (points[:, 0] - c[0]) ** 2 + (points[:, 1] - c[1]) ** 2 < r2
            filter_[u] = filter_[v] = False
            inter = ls[filter_]

            if inter.size == 0:
                self.add([u, v], sqrt(r2))
            for w in inter:
                tup = tuple(sorted([u, v, w]))
                if tup in self.filtration_values:
                    self.add([u, v], self.filtration_values[tup])

        self.points = points

    def get_alpha_r(self, r=inf):
        """
        Return a alpha complex at an instance (radius). That's the simplices with value less than
        filtration_value.

        Args:
            r (float, default math.inf): filtration value

        Returns:
            (AlphaComplex)
        """
        filtration = super().get_by_filtration_value(r)

        alpha = AlphaComplex()
        alpha.filtration_values = filtration.filtration_values
        alpha.simplices_maximales = filtration.simplices_maximales
        alpha.points = np.copy(self.points)

        return alpha

    def persistent_homology(self):
        """

        Args:
            radius:

        Returns:

        """
        matrix, faces = self.edge_matrix()
        dgm = [[(0, inf)], []]

        for j in range(len(faces)):
            while True:
                indices = matrix[:, j].nonzero()[0]
                i = indices[-1] if indices.size != 0 else -1
                if i < 0:
                    break
                else:
                    indices = matrix[i, :j].nonzero()[0]
                    for j0 in indices:
                        if matrix[i:, j0].nonzero()[0].size == 1:
                            break
                    else:
                        if len(faces[j]) == 2:
                            dgm[0].append((0, self.filtration_values[faces[j]]))
                        else:
                            dgm[1].append((self.filtration_values[faces[i]], self.filtration_values[faces[j]]))
                        break

                    matrix[:, j] ^= matrix[:, j0]

        return dgm

    def plot_persistent_homology(self, dgm=None, r=inf, t='d', dpi=300):
        """
        Plots the persitence diagram (t='d') or the bar code (t='b')
        If dgm is None it will call the persistent diagram.

        Args:
            dgm (list(list(tuples)), default None):
            r (float):
            t (str): 'd' for persistence diagram, 'b' for bar code
            dpi (int):
        """

        if not dgm:
            dgm = self.persistent_homology()

        max_ = max(self.filtration_values.values())
        max_ = r if max_ > r else max_ * 1.1

        dgm_filtered = []

        for dgm_ in dgm:
            infs = []
            dgm_filtered.append([])
            j = 0
            for i, tup in enumerate(dgm_):
                if tup[0] <= r:
                    dgm_filtered[-1].append(tup)
                else:
                    continue
                if tup[1] >= r:
                    infs.append((i, j))
                j += 1

            for (i, j) in infs:
                dgm_filtered[-1][j] = (dgm_[i][0], max_)

        plt.figure(dpi=dpi)

        if t == 'd':
            self.__plot_persistent_diagram__(dgm_filtered, max_)
        else:
            self.__plot_persistent_bar_code__(dgm_filtered, max_)

    @staticmethod
    def __plot_persistent_diagram__(dgm, max_):
        """
        Plots the persistent diagram of a given list of birth death pairs (dgm) and the maximum of a filtration
        (max_)

        Args:
            dgm (list(list(tuple))): list of lists of birth death pairs
            max_ (float): maximum
        """
        count = {tup: dgm_.count(tup) for dgm_ in dgm for tup in dgm_}
        for tup in count.keys():
            if count[tup] > 1:
                plt.text(tup[0] + max_ / 55, tup[1] - max_ / 27, str(count[tup]))

        plt.plot((0, max_), (0, max_), '--', color='cornflowerblue', label=r"$\infty$")
        plt.plot((0, max_), (max_, max_), '--', color='cornflowerblue')

        plt.scatter(*zip(*dgm[0]), color='blue', label="$dgm_0$")
        if len(dgm[1]) > 0:
            plt.scatter(*zip(*dgm[1]), color='red', label="$dgm_1$")

        plt.xlabel('Birth Time')
        plt.ylabel('Death Time')
        plt.legend()
        plt.show()

    @staticmethod
    def __plot_persistent_bar_code__(dgm, max_):
        """
        Plots the persistent bar code of a given list of birth death pairs (dgm)

        Args:
            dgm (list(list(tuple))): list of lists of birth death pairs
        """

        for i, tup in enumerate(dgm[0]):
            if i == 0:
                plt.plot((tup[0], tup[1]), (i, i), '-', color='b', linewidth=2, label="$dgm_0$")
            else:
                plt.plot((tup[0], tup[1]), (i, i), '-', color='b', linewidth=2)

        plt.axhline(y=i + 1, color='black', linestyle='--')
        for j, tup in enumerate(dgm[1]):

            if tup[0] == tup[1]:
                plt.scatter([tup[0]], [j + i + 2], s=1, c='r')
            if j == 0:
                plt.plot((tup[0], tup[1]), (j + i + 2, j + i + 2), '-', color='r', linewidth=2, label="$dgm_1$")
            else:
                plt.plot((tup[0], tup[1]), (j + i + 2, j + i + 2), '-', color='r', linewidth=2)

        ax = plt.axes()
        ax.get_yaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.legend()
        plt.show()

    def plot_alpha_r(self, r=inf, cmap=None, dpi=300):
        """
        Plot a alpha complex at an instance (radius) including the Voronoi partition.

        Args:
            r (float, default math.inf): filtration value
            cmap (default None):
            dpi (int):
        """
        vor = Voronoi(self.points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_width=2, line_colors='red')
        fig.dpi = dpi
        self.__plot_alpha_r(r, fig, cmap=cmap)
        plt.show()

    def __plot_alpha_r(self, r, fig, cmap=None):
        """
        Plot a alpha complex at an instance (radius)

        Args:
            r (float): filtration value
            fig (matplotli.pyplot.figure): figure to plot on
            cmap (default None):
        """

        ls_dim2 = []
        ls_dim3 = []

        for simp in self.filtration_values.items():
            if simp[1] <= r:
                if len(simp[0]) == 2:
                    ls_dim2.append(self.points[list(simp[0])])
                elif len(simp[0]) == 3:
                    ls_dim3.append(simp)

        if ls_dim2:
            for edge in ls_dim2:
                plt.plot(edge[:, 0], edge[:, 1], color='black')

        if ls_dim3:
            tris, cs = zip(*ls_dim3)
            tpc = plt.tripcolor(self.points[:, 0], self.points[:, 1], tris, cs, cmap=cmap, edgecolor="k", lw=2)
            if cmap is None or isinstance(cmap, str):
                fig.colorbar(tpc)

        # plt.plot(self.points[:, 0], self.points[:, 1], 'ko')

    def animate(self, interval=100):
        """
        Shows how the complex it's created (in order).
        """
        vor = Voronoi(self.points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_width=2, line_colors='red')
        plt.plot(self.points[:, 0], self.points[:, 1], 'ko')

        frames = sorted(set(self.filtration_values.values()))

        def func(r):
            self.__plot_alpha_r(r, fig=fig, cmap=colors.ListedColormap("limegreen"))

        ani = animation.FuncAnimation(fig, func, frames=frames[1:], interval=interval)
        plt.show()
