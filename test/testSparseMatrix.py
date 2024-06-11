import unittest
import sys
import libpeigen as peigen

class SparseMatrixTest(unittest.TestCase):
    def setUp(self):
        self.rows = 2000
        self.cols = 3000
        self.sparse_matrix = peigen.sparseMatrixDouble(self.rows, self.cols)
        
    def test_rows(self):
        self.assertEqual(self.sparse_matrix.rows(), self.rows)
        
    def test_cols(self):
        self.assertEqual(self.sparse_matrix.cols(), self.cols)
        
    def test_norm(self):
        self.sparse_matrix.setElem(3.14, 5, 4)
        norm_greater_than_0 = (self.sparse_matrix.norm() > 0)
        self.assertEqual(norm_greater_than_0, True)
        
    #def test_integer_multiplication(self):
    #    elem = self.sparse_matrix.getElem(1,1)
    #    self.sparse_matrix *= int(2)
    #    self.assertEqual(self.sparse_matrix.getElem(1,1), int(2)*elem)
        
    #def test_float_multiplication(self):
    #    elem = self.sparse_matrix.getElem(1,1)
    #    self.sparse_matrix *= float(2.0)
    #    self.assertEqual(self.sparse_matrix.getElem(1,1), float(2.0)*elem)
        
    #def test_plus_equals(self):
    #    rhs = peigen.denseMatrixDouble(self.rows, self.cols)
    #    rhs.setRandom(3)
    #    result = self.dense_matrix.getElem(1,1) + rhs.getElem(1,1)
    #    self.dense_matrix += rhs
    #    self.assertEqual(self.dense_matrix.getElem(1,1), result)
        
    #def test_assignment(self):
    #    lhs = peigen.denseMatrixDouble()
    #    lhs.assign(self.dense_matrix)
    #    self.assertEqual(lhs.norm(), self.dense_matrix.norm())
        
    #def test_matrix_multiplication(self):
    #    rhs = peigen.denseMatrixDouble(self.rows, self.cols)
    #    rhs.setRandom(3)
    #    result = self.dense_matrix.transpose()*rhs
    #    norm_greater_than_previous = (result.norm() > self.dense_matrix.norm())
    #    self.assertEqual(norm_greater_than_previous, True)
    #    self.assertEqual(result.rows(), self.dense_matrix.cols())
    #    self.assertEqual(result.cols(), rhs.cols())
        
if __name__ == '__main__':
    unittest.main()