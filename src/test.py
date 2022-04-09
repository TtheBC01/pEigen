import sys
sys.path.append('/pEigen/src/lib')
import libpeigen as peigen

data = peigen.denseMatrixDouble(40,40)
data.setRandom(2)
print(data.rows())
print(data.cols())
print(data.norm())

factorization = peigen.denseDecomposition(data)
factorization.computeThinSVD()
S = factorization.getSingularValues()
print(S.diagonal(0).norm())