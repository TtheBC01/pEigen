# pEigen: Eigen for Python

This project is a simple wrapper for the [Eigen Tux](https://eigen.tuxfamily.org/) library using 
[Boost.Python](https://github.com/boostorg/python). It exposes dense and sparse matrix objects as
well as matrix decomposition methods. Matrix compuatation (particularly factorization) using Eigen
are usually *much* faster that other packages like numpy or scipy.

For information about building this project locally for contributing, see the [developer docs](/DEV.md).

## Examples

Here are a few quick examples of how to use pEigen:

### Dense Matrices

You can create dense matrices by specifying the number of rows and columns. Matrices are stored in column-major order (currently this is the only storage model).

```python
import libpeigen as peigen

rows = 20
cols = 30
dmat = peigen.dense_matrix(rows, cols)

# set the element on the second row in the first column to 4
dmat.set_elem(4,1,0)

# you can get a single element like this
myElement = dmat.get_elem(1,0) # should be 4

# initialize the whole matrix with random double precision floats with a seed value
# if you use the same seed, you will get the same random matrix on the same machine
dmat.set_random(3)

# the [] operator will return a dense matrix object
dmat[1].show() # this will return the second row as a dense matrix
dmat[1][0].show() # this will return the element on the second row on the first column as a sense matrix

print("Number of Rows: ", dmat.rows())
print("Number of Cols: ", dmat.cols())
print("Matrix Norm: ", dmat.norm())

# You can always call .show() on a matrix object to print the contents
# in a pretty layout
dmat.transpose().show()

# scalar multiplication
dmat *= 3.14

# matrix multiplication
new_dmat = peigen.dense_matrix(rows,cols)
new_dmat.assign(dmat) # deep copy into new dense matrix object
result = dmat * new_dmat.transpose() # result is a new dense matrix object

# matrix addition
result = dmat + new_dmat
# or
dmat += new_dmat

# matrix subtraction
result = dmat - new_dmat
# or
dmat -= new_dmat
```

You can also access arbitrary blocks of a dense matrix.

```python
startingRow = 5
startingCol = 6
num_rows = 4
num_cols = 3

# make a new matrix from a block of an existing matrix
my_block = dmat.block(startingRow, startingCol, num_rows, num_cols)

# or grab the (off)diagonals from a matrix
my_diagonal = dmat.diagonal(1) # returns the diagonal of a (potentially rectangular) offset by 1 in this case
```

You can save a dense matrix to a file to use later.

```python
dmat.save("myMat.mat")

new_dmat = peigen.dense_matrix()
new_dmat.load("myMat.mat")
```

### Sparse Matrices

Just like with dense matrices, you can create a sparse matrix by pre-specifying the number of rows and columns. Sparse matrices are stored
in [compressed sparse column](https://docs.nvidia.com/nvpl/_static/sparse/storage_format/sparse_matrix.html#compressed-sparse-column-csc) (CSC) format. 

```python
import libpeigen as peigen

# sparse matrices can be very large yet take up very little memory if the number of non-zero elements is small
rows = 1000
cols = 2000

smat = peigen.sparse_matrix(rows,cols)

smat.nnz() # will be 0 right after initialization

# you can set individual elements like this
smat.set_elem(3.14, 499, 299) # now the element on the 500th row and 300th column is 3.14

# get the number of rows, columns, and the matrix norm just like with dense matrices
print("Number of Rows: ", smat.rows())
print("Number of Cols: ", smat.cols())
print("Matrix Norm: ", smat.norm())

# you can multiply two sparse matrices together
new_smat = peigen.sparse_matrix()
new_smat.assign(smat)
result = new_smat.transpose() * smat

# you can also multiply dense and sparse matrices together
dmat = peigen.dense_matrix(rows, cols)
dmat.set_random(1)
result = dmat.transpose() * smat
result = smat.transpose() * dmat
```

You can also save a sparse matrix to a file to use later.

```python
smat.save("myMat.mat")

new_smat = peigen.dense_matrix()
new_smat.load("myMat.mat")
```

## Factorizations

The pEigen package exposes SVD and QR decomposition from Eigen Tux.

### Singular Value Decomposition

```python
import libpeigen as peigen

rows = 10
cols = 20
dmat = peigen.dense_matrix(rows, cols)
dmat.set_random(1)

# initialize a factorizer object 
factorizer = peigen.factorizer(dmat)

# compute the singular value decomposition (USV^T) with the Bidiagonal Divide and Conquer method
factorizer.bdcsvd()

# get the singular values as a dense diagonal matrix
factorizer.get_singular_values().show()

# get the left singular vector matrix
factorizer.get_u().show()

# get the right singular vector matrix
factorizer.get_v().show()
```

### QR Factorization

```python
rows = 10
cols = 20
dmat = peigen.dense_matrix(rows, cols)
dmat.set_random(1)

# initialize a factorizer object 
factorizer = peigen.factorizer(dmat)

# compute QR decomposition with the Householder method
factorizer.householder_qr()

# get the Q matrix
factorizer.get_q().show()
```