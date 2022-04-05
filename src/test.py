import sys
sys.path.append('./lib')
import libpeigen

data = libpeigen.denseMatrixDouble(4,4)
data.setRandom(2)
print(data.rows())
print(data.cols())
data.show()