# package
import os
import tqdm
import numpy
import pickle
import warnings
warnings.filterwarnings('ignore')
# self
import utils
utils.clear_cache()
from layers import Linear
from loader import mnist_loader
from functional import Sigmoid, Tanh, ReLU, MseLoss




class BPNN(object):
    def __init__(self, dimensions=None, activation=None):
        if dimensions is None:
            dimensions = [784, 100, 10]
        self.dimensions = dimensions
        # 网络权重
        self.layers = [Linear(dimensions[i], dimensions[i + 1], bias=True) for i in range(len(dimensions) - 1)]
        # 激活函数
        self.activation = [Sigmoid() for i in range(len(self.layers))] if activation is None else activation

    def __str__(self):
        structure = ''
        for cnt, (layer, activation) in enumerate(zip(self.layers, self.activation)):
            structure += 'layer {}==>  ({}, {})\n'.format(cnt, layer.in_dim, layer.out_dim)
            structure += 'activation==>  {}\n'.format(activation.__class__.__name__)
        return structure

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activation):
            x = layer(x)
            x = activation(x)
            # layer.output = x
        return x

    def backward(self, loss_fn, learning_rate=1e-3):

        delta = [0] * len(self.layers)

        for i in reversed(range(len(self.layers))):
            # 如果是最后一层
            if i == len(self.layers) - 1:
                error = loss_fn.backward()
                delta[i] = error * self.activation[i].backward()
            else:
                '''
                    中间层, 每一个权重都会影响到下一层的所有神经元; 梯度反向更新时, 需要求和
                        layer  :  (100, 10)
                        delta[1]  :  (10,)
                    等效于以下操作:
                        error = numpy.array([numpy.sum(it * delta[i + 1]) for it in self.layers[i + 1].weights])

                '''
                error = numpy.dot(self.layers[i + 1].weights, delta[i + 1])
                delta[i] = error * self.activation[i].backward()
        # 更新一波权重
        for i in range(len(self.layers)):
            self.layers[i].weights -= learning_rate * self.layers[i].backward().reshape(-1, 1) * delta[i]
            self.layers[i].bias -= learning_rate * delta[i]


    def recognize(self, inpt):
        return numpy.argmax(self.forward(inpt))


    def score(self, x, y):
        guess = [self.recognize(one) for one in x]
        return sum(guess == y) / len(x)



class Solver(object):
    def __init__(self, network, loader):
        self.network = network
        # 损失函数
        self.loss_fn = MseLoss()

        # 加载数据集
        self.loader = loader
        self.loader.load_data()


    def save(self, path=None):
        if path is None: path = './checkpoints/bpnn.pkl'
        with open(path, 'wb') as writer:
            pickle.dump(self.network, writer)
            print('save model to {}'.format(path))

    def load(self, path):
        assert os.path.exists(path), "the path of pretrained model doesn't exist !"
        with open(path, 'rb') as reader:
            self.network = pickle.load(reader)
            print('load model from {}  successfully !'.format(path))


    def train(self, epoches=1, learning_rate=1e-2, shuffle=True):
        for epoch in range(epoches):
            with utils.Timer() as time_scope:
                # 训练一个 epoch
                process_bar = tqdm.tqdm(zip(self.loader.train_images, self.loader.train_labels))
                for cnt, (x, y) in enumerate(process_bar):
                    # 将标签 one-hot 化
                    y = utils.one_hot(y, _len=self.network.dimensions[-1])
                    # 前向计算, 估计得到 10 个数字的概率
                    output = self.network.forward(x)
                    # 计算 MSE 损失
                    loss_value = self.loss_fn(output, y)
                    # 计算梯度, 权重更新
                    self.network.backward(self.loss_fn, learning_rate=learning_rate)
                    # 观察信息; 去掉下面两行可以大大加快训练速度
                    process_bar.set_description('epoch {} | batch {} '.format(epoch + 1, cnt + 1))
                    process_bar.set_postfix(loss=numpy.mean(loss_value))

                # 验证这个 epoch 的效果
                score = self.network.score(self.loader.valid_images, self.loader.valid_labels)
                print('epoch {}====>   accuracy on valid datasets  :  {}'.format(epoch + 1, score))

                # 保存一次模型
                self.save('./checkpoints/bpnn_epoch_{}_accuracy_{}.pkl'.format(epoch + 1, score))

                # 每训练一个 epoch 打乱顺序
                if(shuffle): utils.shuffle_togather([self.loader.train_images, self.loader.train_labels])

        # 测试集上
        score = self.network.score(self.loader.test_images, self.loader.test_labels)
        print('\naccuracy on test datasets  :  {}'.format(score))
       
       


if __name__ == '__main__':

    bpnn = BPNN(dimensions=[784, 100, 10], activation=None)

    data_loader = mnist_loader()

    solver = Solver(network=bpnn, loader=data_loader)

    solver.train(epoches=1, learning_rate=5 * 1e-2)

