import numpy



class Linear():
    def __init__(self, in_dim, out_dim, bias=True):
        self.in_dim, self.out_dim = in_dim, out_dim
        # 随机初始化权重
        self.weights = numpy.random.uniform(-0.5, 0.5, (in_dim, out_dim))
        self.bias = 0 if(bias is False) else numpy.random.randn(out_dim) * 0.1

    def __call__(self, x):
        self.inpt = x
        return numpy.dot(x, self.weights) + self.bias

    def backward(self):
        # 我记得好像要保留一些变量来着, 可以试试; 对 W 求导......
        return self.inpt.reshape(-1, self.in_dim)

if __name__ == '__main__':
    

    layer = Linear(2, 3)

    x = numpy.array([1.5, 2.5])

    result = layer(x)

    print(layer.backward())
