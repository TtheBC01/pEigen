import unittest
import sys
sys.path.append('/pEigen/src/lib')
import libpeigen as peigen

class DenseFactorizationTest(unittest.TestCase):
    def setUp(self):
        self.rows = 10
        self.cols = 10
        
        self.dense_matrix = peigen.denseMatrixDouble(self.rows, self.cols)
        self.dense_matrix.setRandom(1)
        
        self.factorizer = peigen.denseDecomposition(self.dense_matrix)
        
    def test_thin_svd(self):
        self.factorizer.computeThinSVD()
        S = self.factorizer.getSingularValues()
        norm_greater_than_0 = (S.diagonal(0).norm() > 0)
        U = self.factorizer.getU()
        UtU = U.transpose()*U
        trace = UtU.trace()
        residual = (trace - UtU.rows())**2/(UtU.rows()**2)
        res_less_than_eps = (residual < 1e-9)
        self.assertEqual(res_less_than_eps, True)
        self.assertEqual(norm_greater_than_0, True)
        
    def test_qr_decomp(self):
        self.factorizer.computeQR()
        Q = self.factorizer.getQ()
        QtQ = Q.transpose()*Q
        trace = QtQ.trace()
        residual = (trace - QtQ.rows())**2/(QtQ.rows()**2)
        res_less_than_eps = (residual < 1e-9)
        self.assertEqual(res_less_than_eps, True)
 

if __name__ == '__main__':
    unittest.main()