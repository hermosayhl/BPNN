import numpy


# --------------------------------------- 损失函数类 ---------------------------------------------------



class MseLoss():
    def __init__(self):
        pass

    def __call__(self, lhs, rhs, fixed=True):
        if fixed is True:
            self.lhs, self.rhs = lhs, rhs
        else:
            self.lhs, self.rhs = rhs, lhs
        return 1. / 2 * numpy.square(lhs - rhs)

    def backward(self):
        # 这里要注意 __call__ 的位置问题
        return self.lhs - self.rhs



# --------------------------------------- 激活函数类 ---------------------------------------------------

class Sigmoid():
    def __init__(self):
        pass

    def __call__(self, x):
        self.output = 1.0 / (1.0 + numpy.exp(-x))
        return self.output

    def backward(self):
        return self.output * (1.0 - self.output)




class Tanh():
    def __init__(self):
        pass

    def __call__(self, x):
        a, b = numpy.exp(x), numpy.exp(-x)
        self.output = (a - b) / (a + b)
        return self.output

    def backward(self):
        return 1. - self.output ** 2



# 暂时有问题
class ReLU():
    def __init__(self, alpha=None):
        self.alpha = alpha

    def __call__(self, x):
        if self.alpha is None:
            self.x = x
            x[x < 0] = 0
            return x
        # α 不为 None 时, 就是 Leaky ReLU

    def backward(self):
        self.x[self.x > 0] = 1
        self.x[self.x < 0] = 0
        return self.x





# ----------------------------------------------------------------------------------------

class SoftMax():
    def __init__(self):
        pass

    def __call__(self, lhs, rhs):
        pass