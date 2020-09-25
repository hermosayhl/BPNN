import os
import gzip
import numpy
import random
import struct
import _pickle as cPickle
import pickle



class mnist_loader():
	def __init__(self):
		pass

	def load_data(self, src='./datasets/mnist.pkl.gz'):
		if hasattr(self, 'train_data'): 
			return
		assert os.path.exists(src), "the source file {} doesn't exists !".format(src)
		# 直接读取, 文件和读取方法直接参考 https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
		with gzip.open(src, 'rb') as reader:
			train, valid, test = cPickle.load(reader, encoding='bytes')

			self.train_images = train[0]
			self.train_labels = train[1]
			self.valid_images = valid[0]
			self.valid_labels = valid[1]
			self.test_images = test[0]
			self.test_labels = test[1]

			print('train==>  {}\nvalid==>  :  {}\ntest==>  :  {}'.format(len(self.train_images), len(self.valid_images), len(self.test_images)))


	# 默认只需要对训练集打乱
	def mini_batch(self, batch_size=16, shuffle=True):
		L = len(self.train_images)
		batch_num = int(L / batch_size)

		for cnt in range(batch_num):
			if(shuffle == True):
				indexes = [random.randint(0, L - 1) for i in range(batch_size)]
				yield numpy.array([self.train_images[idx] for idx in indexes]),\
					  numpy.array([self.train_labels[idx] for idx in indexes])


	# 展示若干图像
	def show_some_images(self, scope='train'):
		import math
		import matplotlib.pyplot as plt

		images = getattr(self, scope + '_images')
		labels = getattr(self, scope + '_labels')
		# 随机抽取 16 张图象
		indexes = [random.randint(0, len(images)) for i in range(16)]
		images = [images[idx] for idx in indexes]
		labels = [labels[idx] for idx in indexes]
		# 设置画布
		e = int(math.sqrt(len(images)))
		plt.figure(figsize=(16, 16))
		for j in range(len(images)):
			plt.subplot(e, e, j + 1)
			plt.xticks([])
			plt.yticks([])
			title = u"label  " + str(labels[j])
			plt.title(title)
			plt.imshow(images[j].reshape(28, 28), cmap='gray')
		plt.savefig('./number_{}.png'.format(scope))
		plt.show()
		
			




# refer to the code in https://www.jianshu.com/p/81f8ca1b722
# 直接读取原始数据, 需要归一化到 0-1
class mnist_loader_2():
	'''
		self.train_loader = loader.mnist_loader_2(
            images_path='../../../datasets/Mnist/train-images.idx3-ubyte',
            labels_path='../../../datasets/Mnist/train-labels.idx1-ubyte',
            dataset_name='train_for_mnist',
            pre_load_path='./preload/train_for_mnist.pkl'
        )
        self.test_loader = loader.mnist_loader_2(
            images_path='../../../datasets/Mnist/t10k-images.idx3-ubyte',
            labels_path='../../../datasets/Mnist/t10k-labels.idx1-ubyte',
            dataset_name='test_for_mnist',
            pre_load_path='./preload/test_for_mnist.pkl'
        )
	'''
	def __init__(self, images_path, labels_path, pre_load_path='./train_for_mnist.pkl', dataset_name='train', image_len=784, label_len=1, _size=(28, 28)):
		
		self.dataset_name = dataset_name

		# 检查是否需要 预加载
		if(os.path.exists(pre_load_path)):
			with open(pre_load_path, 'rb') as reader:
				images_and_labels = pickle.load(reader)
				self.images, self.labels = images_and_labels['images'], images_and_labels['labels']
				self.images_num = len(self.images)
		else:
			# 读取到缓冲文件
			with open(images_path, 'rb') as reader:
				images_buffer = reader.read()
			with open(labels_path, 'rb') as reader:
				labels_buffer = reader.read()

			# 读取标签头
			images_index = 0 + struct.calcsize('>IIII')
			labels_index = 0 + struct.calcsize('>II')

			# 计算图像个数
			self.images_num = int((len(images_buffer) - images_index) / image_len)
			assert self.images_num == int((len(labels_buffer) - labels_index) / label_len), "images and labels for {} is not the same".format(self.dataset_name)
			
			print('Total  {}  images in dataset  {}'.format(self.images_num, self.dataset_name))

			# 读取图像和标签
			self.images = []
			self.labels = []
			for cnt in range(self.images_num):
				# images
				temp = struct.unpack_from('>{}B'.format(image_len), images_buffer, images_index) # 从位置 images_index 开始读取 image_len 长度的内容
				images_index += struct.calcsize('{}B'.format(image_len)) # 移到下一次读取的开始位置, 偏移 784 
				self.images.append(numpy.reshape(temp, (image_len)))
				# labels
				temp = struct.unpack_from('>{}B'.format(label_len), labels_buffer, labels_index)
				labels_index += struct.calcsize('{}B'.format(label_len))
				self.labels.append(temp[0])

			print('{}  :  {} images  and  {}  labels'.format(self.dataset_name, numpy.array(self.images).shape, numpy.array(self.labels).shape))

			self.images = numpy.array(self.images) * 1. / 255 
			self.labels = numpy.array(self.labels)

			if(not os.path.exists(pre_load_path)):
				with open(pre_load_path, 'wb') as writer:
					pickle.dump({'images': self.images, 'labels': self.labels}, writer)



			


if __name__ == '__main__':

	loader = mnist_loader()

	loader.load_data()

	loader.show_some_images()
	