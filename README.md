# computational_topology

An implementation in Python of a simplicial complex, a filtration, an Alpha complex and Vietoris-Rips complex.

---
## SimplicialComplex

A simplicial complex is a set os points, segments, ..., n-dimensional simplices.

### Betti nums

Used to distinguish topological spaces, they can also be used for simplicial complexes.
For example:

```python
sc = SimplicialComplex()
sc.load('botella de Klein')
print('The Betti numbers of the Klein bottle are: ', sc.betti_nums())
```

---
## Filtration


A filtration is a sequence of a simplicial complexes.

---
## AlphaComplex

An AlphaComplex is a filtration determined by the Delaunay triangulation.

### persistent homology
 
You can get the persistent diagram (dgm) by doing:


```python
alpha = AlphaComplex(points=points)
dgm = alpha.persistent_homology()
```

or plot the persistent diagram:

````python
alpha = AlphaComplex(points=points)
dgm = alpha.persistent_homology()
alpha.plot_persistent_homology(dgm=dgm, t='d')
````

or plot the persistent bar code:

````python
alpha = AlphaComplex(points=points)
dgm = alpha.persistent_homology()
alpha.plot_persistent_homology(dgm=dgm, t='b')
````

