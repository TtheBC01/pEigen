import unittest
import random
import libpeigen as peigen

def fill_sparse_matrix(matrix, sparsity_percentage):
    """
    Fills a given sparse matrix with random float values at random positions
    based on the target sparsity percentage.

    Parameters:
    matrix (sparseMatrixDouble): The empty sparse matrix to fill 
    sparsity_percentage (float): The target sparsity percentage (0-100), keep it to single digits).

    Returns:
    row_indices: list of row indices of non-zero elements
    col_indices: list of column indices of non-zero elements
    data: list of element values on non-zero elements
    """
    rows = matrix.rows()
    cols = matrix.cols()
    total_elements = rows*cols
    nnz = int(total_elements * (sparsity_percentage/100))

    # Generate random positions for the non-zero elements
    row_indices = [random.randint(0, rows - 1) for _ in range (nnz)]
    col_indices = [random.randint(0, cols - 1) for _ in range (nnz)]

    # Zip the rows and columns into a set to avoid duplicates
    index_set = set(zip(row_indices, col_indices))

    # Generate random float values for the nz elements
    data = [random.random() for _ in range(len(index_set))]

    # zip the values with the indices
    elements = set(zip(index_set, data))

    # now set each element in the sparse matrix
    for elem in elements:
        matrix.setElem(elem[1], elem[0][0], elem[0][1])

    return elements

class SparseMatrixTest(unittest.TestCase):
    def setUp(self):
        self.rows = 1000
        self.cols = 2000
        self.sparse_matrix = peigen.sparseMatrixDouble(self.rows, self.cols)
        self.elements = fill_sparse_matrix(self.sparse_matrix, 2)

    def test_assignment(self):
        lhs = peigen.sparseMatrixDouble()
        lhs.assign(self.sparse_matrix)
        self.assertEqual(lhs.norm(), self.sparse_matrix.norm())
        
    def test_rows(self):
        self.assertEqual(self.sparse_matrix.rows(), self.rows)
        
    def test_cols(self):
        self.assertEqual(self.sparse_matrix.cols(), self.cols)

    def test_set_element(self):
        for elem in self.elements:
            self.assertEqual(elem[1], self.sparse_matrix.getElem(elem[0][0], elem[0][1]))
        
    def test_norm(self):
        norm_greater_than_0 = (self.sparse_matrix.norm() > 0)
        self.assertEqual(norm_greater_than_0, True)
        
    def test_integer_multiplication(self):
        sm = peigen.sparseMatrixDouble()
        sm.assign(self.sparse_matrix)
        sm *= int(2)
        self.assertEqual(self.sparse_matrix.norm()*int(2), sm.norm())
        
    def test_float_multiplication(self):
        sm = peigen.sparseMatrixDouble()
        sm.assign(self.sparse_matrix)
        sm *= float(2)
        self.assertEqual(self.sparse_matrix.norm()*float(2), sm.norm())
        
    def test_plus_equals(self):
        rhs = peigen.sparseMatrixDouble(self.rows, self.cols)
        rhs.assign(self.sparse_matrix)
        self.sparse_matrix += rhs
        self.assertEqual(self.sparse_matrix.norm(), int(2)*rhs.norm())
        
    def test_sparse_sparse_matrix_multiplication(self):
        rhs = peigen.sparseMatrixDouble(self.rows, self.cols)
        fill_sparse_matrix(rhs, 4)
        result = self.sparse_matrix.transpose()*rhs
        result_norm_leq_product = (result.norm() <= (self.sparse_matrix.norm() * rhs.norm()))
        self.assertEqual(result_norm_leq_product, True)
        self.assertEqual(result.rows(), self.sparse_matrix.cols())
        self.assertEqual(result.cols(), rhs.cols())

    def test_sparse_dense_matrix_multiplication(self):
        rhs = peigen.denseMatrixDouble(self.rows, self.cols)
        rhs.setRandom(1)
        result = self.sparse_matrix.transpose()*rhs
        result_norm_leq_product = (result.norm() <= (self.sparse_matrix.norm() * rhs.norm()))
        self.assertEqual(result_norm_leq_product, True)
        self.assertEqual(result.rows(), self.sparse_matrix.cols())
        self.assertEqual(result.cols(), rhs.cols())

    def test_dense_sparse_matrix_multiplication(self):
        rhs = peigen.denseMatrixDouble(self.rows, self.cols)
        rhs.setRandom(1)
        result = rhs.transpose()*self.sparse_matrix
        result_norm_leq_product = (result.norm() <= (self.sparse_matrix.norm() * rhs.norm()))
        self.assertEqual(result_norm_leq_product, True)
        self.assertEqual(result.rows(), self.sparse_matrix.cols())
        self.assertEqual(result.cols(), rhs.cols())
        
if __name__ == '__main__':
    unittest.main()