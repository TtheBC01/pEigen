# pEigen: Eigen for Python

This project is a simple wrapper for the [Eigen Tux](https://eigen.tuxfamily.org/) library using 
[Boost.Python](https://github.com/boostorg/python). It exposes dense and sparse matrix objects as
well as matrix decomposition methods. 


For information about building this project locally for contributing, see the [developer docs](/DEV.md).

## Examples

Here are a few quick examples of how to use pEigen:

### Dense Matrices

You can create dense matrices by specifying the number of rows and columns. Elements can be set individually using `setElem` and the Frobenius norm can be calculated by calling `.norm` on the object.

```python
import libpeigen as peigen

rows = 20
cols = 30
denseMat = peigen.denseMatrixDouble(rows, cols)

# set the element on the second row in the first column to 4
denseMat.setElem(4,1,0)

# initialize the whole matrix with random double precision floats with a seed value
# if you use the same seed, you will get the same random matrix on the same machine
denseMat.setRandom(3)

print("Number of Rows: ", denseMat.rows())
print("Number of Cols: ", denseMat.cols())
print("Matrix Norm: ", denseMat.norm())

# You can always call .show() on a dense matrix object to print the contents
# in a pretty layout
denseMat.transpose().show()

# scalar multiplication
denseMat *= 3.14

# matrix multiplication
newDenseMat = peigen.denseMatrixDouble(rows,cols)
newDenseMat.assign(denseMat) # deep copy into new dense matrix object
result = denseMat * newDenseMat.transpose() # result is a new dense matrix object

# matrix addition
result = denseMat + newDenseMat
# or
denseMat += newDenseMat

# matrix subtraction
result = denseMat - newDenseMat
# or
denseMat -= newDenseMat
```

You can also access arbitrary blocks of a dense matrix.

```python
startingRow = 5
startingCol = 6
numRows = 4
numCols = 3

# make a new matrix from a block of an existing matrix
myBlock = denseMat.block(startingRow, startingCol, numRows, numCols)

# or grab the (off)diagonals from a matrix
myDiagonal = denseMat.diagonal(1) # returns the diagonal of a (potentially rectangular) offset by 1 in this case
```

## Factorizations

The pEigen package exposes SVD and QR decomposition from Eigen Tux.

### Singular Value Decomposition

```python
import libpeigen as peigen

rows = 10
cols = 20
denseMat = peigen.denseMatrixDouble(rows, cols)
denseMat.setRandom(1)

# initialize a factorizer object 
factorizer = peigen.denseDecomposition(denseMat)

# compute the singular value decomposition with the Bidiagonal Divide and Conquer method
factorizer.BDCSVD()

# get the singular values
factorizer.getSingularValues().show()

# get the left singular vector matrix
factorizer.getU().show()

# get the right singular vector matrix
factorizer.getV().show()
```

### QR Factorization

```python
rows = 10
cols = 20
denseMat = peigen.denseMatrixDouble(rows, cols)
denseMat.setRandom(1)

# initialize a factorizer object 
factorizer = peigen.denseDecomposition(denseMat)

# compute QR decomposition with the Householder method
factorizer.HouseholderQR()

# get the Q matrix
factorizer.getQ().show()
```