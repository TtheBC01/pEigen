import unittest
import sys
import libpeigen as peigen

class SparseMatrixTest(unittest.TestCase):
    def setUp(self):
        self.rows = 2000
        self.cols = 3000
        self.sparse_matrix = peigen.sparseMatrixDouble(self.rows, self.cols)
        self.sparse_matrix.setElem(5.0,345,2854)

    def test_assignment(self):
        lhs = peigen.sparseMatrixDouble()
        lhs.assign(self.sparse_matrix)
        self.assertEqual(lhs.norm(), self.sparse_matrix.norm())
        
    def test_rows(self):
        self.assertEqual(self.sparse_matrix.rows(), self.rows)
        
    def test_cols(self):
        self.assertEqual(self.sparse_matrix.cols(), self.cols)
        
    def test_norm(self):
        self.sparse_matrix.setElem(3.14, 5, 4)
        norm_greater_than_0 = (self.sparse_matrix.norm() > 0)
        self.assertEqual(norm_greater_than_0, True)
        
    def test_integer_multiplication(self):
        self.sparse_matrix.setElem(30,2,0)
        sm = peigen.sparseMatrixDouble()
        sm.assign(self.sparse_matrix)
        sm *= int(2)
        self.assertEqual(self.sparse_matrix.norm()*int(2), sm.norm())
        
    def test_float_multiplication(self):
        self.sparse_matrix.setElem(30,2,0)
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
        rhs.assign(self.sparse_matrix)
        result = self.sparse_matrix.transpose()*rhs
        norm_greater_than_previous = (result.norm() > self.sparse_matrix.norm())
        self.assertEqual(norm_greater_than_previous, True)
        self.assertEqual(result.rows(), self.sparse_matrix.cols())
        self.assertEqual(result.cols(), rhs.cols())

    def test_sparse_dense_matrix_multiplication(self):
        rhs = peigen.denseMatrixDouble(self.rows, self.cols)
        rhs.setRandom(1)
        result = self.sparse_matrix.transpose()*rhs
        norm_greater_than_previous = (result.norm() > self.sparse_matrix.norm())
        self.assertEqual(norm_greater_than_previous, True)
        self.assertEqual(result.rows(), self.sparse_matrix.cols())
        self.assertEqual(result.cols(), rhs.cols())

    def test_dense_sparse_matrix_multiplication(self):
        rhs = peigen.denseMatrixDouble(self.rows, self.cols)
        rhs.setRandom(1)
        result = rhs.transpose()*self.sparse_matrix
        norm_greater_than_previous = (result.norm() > self.sparse_matrix.norm())
        self.assertEqual(norm_greater_than_previous, True)
        self.assertEqual(result.rows(), self.sparse_matrix.cols())
        self.assertEqual(result.cols(), rhs.cols())
        
if __name__ == '__main__':
    unittest.main()