import unittest
import sys
import libpeigen as peigen

class factorizerTest(unittest.TestCase):
    def setUp(self):
        self.rows = 1000
        self.cols = 1000
        
        self.dense_matrix = peigen.dense_matrix(self.rows, self.cols)
        self.dense_matrix.set_random(1)
        
        self.factorizer = peigen.factorizer(self.dense_matrix)
        
    def test_thin_svd(self):
        self.factorizer.bdcsvd()
        S = self.factorizer.get_singular_values()
        norm_greater_than_0 = (S.diagonal(0).norm() > 0)
        U = self.factorizer.get_u()
        UtU = U.transpose()*U
        trace = UtU.trace() # should be identity matrix
        residual = (trace - UtU.rows())**2/(UtU.rows()**2) 
        res_less_than_eps = (residual < 1e-9)
        self.assertEqual(res_less_than_eps, True)
        self.assertEqual(norm_greater_than_0, True)
        
    def test_qr_decomp(self):
        self.factorizer.householder_qr()
        Q = self.factorizer.get_q()
        QtQ = Q.transpose()*Q
        trace = QtQ.trace()
        residual = (trace - QtQ.rows())**2/(QtQ.rows()**2)
        res_less_than_eps = (residual < 1e-9)
        self.assertEqual(res_less_than_eps, True)
 

if __name__ == '__main__':
    unittest.main()