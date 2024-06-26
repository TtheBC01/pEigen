import unittest
import sys
import libpeigen as peigen

class DenseMatrixTest(unittest.TestCase):
    def setUp(self):
        self.rows = 2000
        self.cols = 3000
        self.dense_matrix = peigen.dense_matrix(self.rows, self.cols)
        self.dense_matrix.set_random(1)

    def test_list_initialization(self):
        data = self.dense_matrix.to_list()
        lhs = peigen.dense_matrix(data, self.rows, self.cols)
        self.assertEqual(lhs.norm(), self.dense_matrix.norm())
        
    def test_rows(self):
        self.assertEqual(self.dense_matrix.rows(), self.rows)
        
    def test_cols(self):
        self.assertEqual(self.dense_matrix.cols(), self.cols)
        
    def test_col(self):
        self.assertEqual(self.dense_matrix.col(1).rows(), self.rows)

    def test_row(self):
        self.assertEqual(self.dense_matrix.row(1).cols(), self.cols)
        
    def test_norm(self):
        norm_greater_than_0 = (self.dense_matrix.norm() > 0)
        self.assertEqual(norm_greater_than_0, True)
        
    def test_elem_access(self):
        self.dense_matrix.set_elem(3.14, 5, 4)
        self.assertEqual(self.dense_matrix.get_elem(5,4), 3.14)
        
    def test_integer_multiplication(self):
        elem = self.dense_matrix.get_elem(1,1)
        self.dense_matrix *= int(2)
        self.assertEqual(self.dense_matrix.get_elem(1,1), int(2)*elem)
        
    def test_float_multiplication(self):
        elem = self.dense_matrix.get_elem(1,1)
        self.dense_matrix *= float(2.0)
        self.assertEqual(self.dense_matrix.get_elem(1,1), float(2.0)*elem)
        
    def test_plus_equals(self):
        rhs = peigen.dense_matrix(self.rows, self.cols)
        rhs.set_random(3)
        result = self.dense_matrix.get_elem(1,1) + rhs.get_elem(1,1)
        self.dense_matrix += rhs
        self.assertEqual(self.dense_matrix.get_elem(1,1), result)
        
    def test_assignment(self):
        lhs = peigen.dense_matrix()
        lhs.assign(self.dense_matrix)
        self.assertEqual(lhs.norm(), self.dense_matrix.norm())
        
    def test_matrix_multiplication(self):
        rhs = peigen.dense_matrix(self.rows, self.cols)
        rhs.set_random(3)
        result = self.dense_matrix.transpose()*rhs
        norm_greater_than_previous = (result.norm() > self.dense_matrix.norm())
        self.assertEqual(norm_greater_than_previous, True)
        self.assertEqual(result.rows(), self.dense_matrix.cols())
        self.assertEqual(result.cols(), rhs.cols())

    def test_dense_sparse_addition(self):
        dm = peigen.dense_matrix(100,100)
        dm.set_random(1)
        sm = peigen.sparse_matrix(100,100)
        result = dm + sm
        self.assertEqual(result.norm(), dm.norm())

    def test_dense_sparse_subtraction(self):
        dm = peigen.dense_matrix(100,100)
        dm.set_random(1)
        sm = peigen.sparse_matrix(100,100)
        result = dm - sm
        self.assertEqual(result.norm(), dm.norm())
        
if __name__ == '__main__':
    unittest.main()