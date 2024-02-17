<hr/>

### Singular vs Non-singular systems and determinant

<hr/>

1. Singular vs Non-singular systems of equation/matrix. Each equation in the system is a row vector in the matrix representing the sytem. If the system has no solution or infinitely many solutions it's called a singular system/matrix. A singular system has linearly dependent equations (i.e. row vectors are linear combinations of one another).
2. Determinant of a matrix: diag(X) - anti-diag(X) = 0, then matrix is singular. For a NxN matrix, products of all diagonals to the right - products of all diagonals to the left. The diagonals wrap around. For example:

```
A =
[
    [1, 1, 1],
    [3, 2, 1],
    [4, 5, 2]
]

i.e. the upper right diagonal 1 wraps around to 3 and 5, the upper diagonal 1,1 wraps around to 4.

det(A) = [1 * 2 * 2 + 1 * 1 * 4 + 1 * 3 * 5] - [1 * 2 * 4 + 3 * 5 * 1 + 4 * 1 * 1] = [4 + 4 + 15] - [8 + 15 + 4] = 31 - 27 = 4

OR

| a b c |
| d e f |
| g h i |

a * [(e,f), (h,i)] -b * [(d,f), (g,i)] + c * [(d,e), (g,h)]
```

Upper triangular matrices (i.e. bottom triangle is all 0s) have det(A) = diag(A)

Operations that do not change singularity:

- Multiplying a row by a nonzero scalar.
- Adding a row to another row.
- Switching rows.

<hr/>

### Upper triangular, lower triangular, and echelon form matrices

<hr/>

A rectangular matrix is in echelor form (or row echelon form) if it has the following 3 properties:

1. All nonzero rows are above any rows of all zeros.
2. Each leading entry of a row (also called pivot) is in a column to the right of the leading entry of the row above it. (i.e. first non-zero entry if any).
3. All entries in a column below a leading entry are zeros.

If a matrix in echelon form satisfies the following additional conditions, then it is in reduced echelon form (or reduced row echelon form):

4. The leading entry in each nonzero row is 1.
5. Each leading 1 is the only nonzero entry in its column.

![Reduced Row-Echelon Form](./img/reduce-row-echelon-form.png)

Row echelon form (diagonal can be 1s or 0s)

```
[
(1 * * * *)
(0 1 * * *)
(0 0 0 0 0)
(0 0 0 0 0)
(0 0 0 0 0)
]
```

```
Echelon form, but not triangular because not a squared matrix:

[
(1, 2, 3)
(0, 4, 5)
]

Triangular, but not echelon (because the leading entry 5 is not to the right of leading entry 4):

[
(1, 2, 3)
(0, 0, 4)
(0, 0, 5)
]

[
(5,1)
(4,-3)
]

==> to row-echelon form (upper diagonal matrix)

[
(1,0.2)
(0,1)
]

==> to reduced row-echelon form
[
(1,0)
(0,1)
]
```

<hr/>

### Rank of a matriex

<hr/>

In machine learning used for image compression.
Singular Value Decomposition (SVD) changes the rank of a matrix while modifying it as little as possible.

`Rank = N - (Dimension of solution space)`

For example, a non-singular matrix has 1 point as the solution so the solution space is 0.
For a 5x5 matrix we have 5 linearly independet equations so the rank of the matrix would be Rank(A) = 5 - 0 = 5.
For a singular matrix with infinite solutions we have a line which has solution space = 1 or solution space = 2 if the infinitely many solutions span a plane (or higher dimensions depending on the dimensions of the matrix).

A matrix is non-singular only if it has `Full Rank (Rank = number of rows/number of equations)`. This means that the system carries the max amount of information possible (each equation is linearly independent).

The rank is the largest number of linearly independent rows/columns in the matrix.
The relation between rank and row-echelon form matrices is the count of 1s in the diagonal of the matrix.
For example, for a 2x2 matrix:

- Matrix in row-echelon form with rank 2 (non-singular)
  1 0.2
  0 1

- Matrix in row-echelon form with rank 1 (singular)
  1 0.2
  0 0

- Matrix in row-echelon form with rank 0 (singular)
  0 0
  0 0

So rank of a matrix is the number of pivots (non-zero leading entries in a row) for the row-echelon form of the matrix.

![Matrix Rank](./img/matrix-rank.png)

<hr/>

### Vectors, Linear transformations

<hr/>

L1-norm: `|(a,b)| = |a| + |b|`.

L2-norm: `|(a,b)| = sqrt(pow(a,2) + pow(b,2))`.

Generally, Lp-norm of a vector x is:

`||x|| = pow( sum( pow(|x[i]|,p) ), 1/p )`

![Norm-1](./img/vector-norm-1.png)

![Norm-2](./img/vector-norm-2.png)

Norms are useful to know the distance between vectors. In machine learning this is used to get similarities between datapoints expressed by n-features in an n-dimensional space.

Norm of a vector using the dot-product of the vector with itself: `L2-norm = sqrt(dotprod(u, u))`
Two vectors are `orthogonal` if `dotprod(u,v) = 0`.
If we have two vectors, u and v, we can project a vector on the other and compute the dotproduct with the projection (u' or v'): `dotprod(u,v) = |u'| * |v|`.

![Dot product](./img/dotprod.png)

![Geometric Dot product](./img/geom-dotprod.png)

Linear transformations can be seen as matrices (functions) that map vectors to another plane.
For example:

```
Matrix:
(3,1)
(1,2)

Vectors: (0,0) (1,0) (0,1) (1,1)

If we take the dot product of the column vectors and the matrix we get the following new vectors:
(0,0) (3,1) (1,2) (4,3)
```

Linear tranformations are a change of coordinates (basis) after applying the matrix (function) mapping.
If we have the linear transformations (i.e. the N points in the first R^N mapped to the second R^N) then we can do the inverse operation and find the matrix that produces that linear transformation. For example, in R^2 we just need the vector (0,1) and (1,0) and the corresponding mappings (i.e. (2,3) and (3,-1)) to get to the matrix (i.e. [(3,2),(-1,3)] by taking the dotproduct with (1,0) and (0,1)).

Matrix multiplication can be also seen as combining 2 linear transformations into a 3rd one.
The first linear transformation to map to the second R^2 and the second linear transformation to map to the third R^2 gives us the two matrices to multiply to get the third linear transformation to map the first R^2 to the third R^2.
Since in the first R^2 we have the column vectors (1,0) and (0,1), the resulting column vectors in the third R^2 are the two column vectors of the product matrix. The two operands (matrices) need to be switched to get to the product matrix with this linear transformation (i.e linear_transform_2 \* linear_transform_1 = linear_transform_3) as matrix multiplication is not commutative.

![Linear Transformation Matrix Mult](./img/linear-transform-comb.png)
![Matrix Mult Dot Product](./img/matrix-mult-dotprod.png)

`The Identity Matrix (I)` is the matrix that when applied as a linear transformation it does not alter the resulting R^N basis.
The resulting vector is the same as the starting vector when multiplied by I.

`The Inverse Matrix`:

![Identity and Inverse Matrix](./img/matrix-inverse-1.png)
![Identity and Inverse Matrix](./img/matrix-inverse-2.png)

We can find the Inverse Matrix by solving the system of equations where the resulting column vectors are (1,0) and (0,1) (i.e. the second linear transformation results in the basis to be the same as the initial one).

![Identity and Inverse Matrix](./img/matrix-inverse-3.png)

Non-singular matrices ALWAYS have an Inverse Matrix. Singular matrices do NOT (Non-invertible).

<hr/>

### More on determinants

<hr/>

`Singular Transformation` is when the change of basis does not span the entire R^N in the new projection.
For example:

```
Linear transformation (matrix):
|1 1|
|2 2|

Take the vector operation vector multiplied by matrix A to get the new vector: `bA = b'`
(0,0) --> (0,0)
(1,0) --> (1,2)
(0,1) --> (1,2)
(1,1) --> (2,4)

This gives us a degenerate parallelogram, a line going through the origin and with slope = 2 (i.e. y/x).
```

This is due to the fact that the matrix used for the linear transformation is singular (row 2 is a linear combination of row 1).
With a matrix that is singular and with infinitely many solutions (i.e. matrix [(0,0),(0,0)]) the result of the linear transformation would be a point instead of a line.

The `Rank` can be also computed as the dimension of the image resulting from the linear transformation when applying the matrix.
If the matrix is in R^2 and non-singular, it will have rank = 2 and it will result in a projected image of dimension = 2 (i.e. spanning the whole R^2).

The `Determinant` of the matrix is the `Area` of the image when mapping the unit N-dimensional vector to the new basis.
For example:

```
Fundamental basis form in R^2 is the square with coordinates (1,0), (0,1), (1,1), (0,0).
If we apply the matrix [(3,1),(1,2)] which has Determinat = 5 we get a parallelogram with Area = 5 with the following vertices.

(0,0) --> (0,0)
(1,0) --> (3,1)
(0,1) --> (1,2)
(1,1) --> (4,3)
```

If the matrix for the linear transformation is singular it will have Det = 0 and also the area of the resulting image will be 0 (i.e. a line in R^2).

If the determinant is negative we can say the area is negative if we take the component vectors in counter-clockwise order and the relation between the determinant and the area holds.

Also `det(AB) = det(A) * det(B)`, the determinant of the product matrix is equal to the determinant of the first matrix multiplied by the determinant of the second matrix. With this we can always tell that the product matrix, if at least one of the operand matrices is singular, will be singular.

The deteterminant of the Inverse Matrix of `A` is equal to `1 / det(A)`.
Since 0 has no inverse (division by 0 is undefined), a singular matrix has no Inverse Matrix as we've seen previously.

![Determinant of Inverse Matrix](./img/determinant-of-inverse-matrix.png)

It follows that `det(I) = 1`, the determinant of the Identity Matrix is always 1 (only diagonal to multiply which has only 1s).

In Machine Learning a way to get more data is transforming (augmenting) the data that we already have by applying linear transformations, such as rotation and shear (doing matrix multiplications).

<hr/>

### Bases, Eigenvalues, Eigenvectors

<hr/>

A basis is formed by N-vectors originating from the origin in R^N that can span R^N.
For example, in R^2 a set of vectors with the same direction is not a basis.
A basis can be also defined by a `minimal spanning set`. For example, 3 vectors in R^2, even if the spane the whole plane, are not a basis because we need exactly 2 vectors in this case (linearly independent, i.e. not spanning only a line or a point).

The number of elements in the basis is the dimension N in R^N.

A linear transformation can also be seen as a change of bases.

```
A set B of vectors in a vector space V is called a basis (pl.: bases) if every element of V may be written in a unique way as a finite linear combination of elements of B. The coefficients of this linear combination are referred to as components or coordinates of the vector with respect to B. The elements of a basis are called basis vectors.

Equivalently, a set B is a basis if its elements are linearly independent and every element of V is a linear combination of elements of B. In other words, a basis is a linearly independent spanning set (i.e. set of all linear combinations of the vectors in S).
```

![Span](./img/span-1.png)

![Span](./img/span-2.png)

![Span](./img/span-3.png)

`Eigenbases` are particularly useful in PCA (Principal Component Analysis).
An eigenbasis is a basis where every vector is an eigenvector.

The set of all eigenvectors of T corresponding to the same eigenvalue, together with the zero vector, is called an eigenspace, or the characteristic space of T associated with that eigenvalue. If a set of eigenvectors of T forms a basis of the domain of T, then this basis is called an eigenbasis. An eigenbasis is a basis of R^N consisting of eigenvectors of A. Eigenvectors and Linear Independence. Eigenvectors with different eigenvalues are automatically linearly independent. If an n × n matrix A has n distinct eigenvalues then it has an eigenbasis.

For example, the "stretching" factors are called an eigenvalues (3 and 2) and the two vectors in this basis are the eigenvectors.

![Eigenbasis](./img/eigenbasis.png)

To find `eigenvalues` we want to find the matrix to subtract to our linear trasformation matrix in order to have a singular transformation (i.e. the determinant of the resulting difference matrix is 0)

If lambda is an eigenvalue in the following example, then the transformations by applying the matrix and the transformation by scaling the plane by a factor of lambda are equal for infinitely many vectors (x,y).

![Eigenvalues](./img/eigenvalues.png)

The roots (i.e. where the characteristic polynomial is 0) are the eigenvalues.

To find the corresponding eigenvectors we solve the linear system of equations with the eigenvalues (which will yield infinitely many solutions for (x,y) because the eigenvector is one vector that form a basis that span the space R^N).

(1,0) is the eigenvector corresponding to the eigenvalue 2 and (1,1) is the eigenvector corresponding to the eigenvalue 3.

![Eigenvectors](./img/eigenvectors.png)

The relation between the matrix and its eigenvalues and eigenvectors is as follows.

The eigenvectors are the only spaces that get mapped to themselves which are also called the eigenspaces (i.e. the red and blue lines in this example)

![Eigenvectors](./img/eigenvectors-2.png)

![Eigenvectors](./img/eigenvectors-3.png)

In the case of a shear transformation we have only 1 eigenvector and a generalized eigenvector and their relation is as follows.

![Eigenvectors](./img/eigenvector-generalized.png)

An application of eigenvectors is in Principal Component Analysis in Machine Learning (PCA).

![PCA](./img/pca.png)

https://metric.ma.ic.ac.uk/metric_public/matrices/eigenvalues_and_eigenvectors/eigenvalues2.html
