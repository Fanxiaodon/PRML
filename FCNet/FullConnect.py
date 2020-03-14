import numpy as np


class SigmoidActivator(object):
    # sigmoid函数前向计算输出
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    # 用于反向计算误差\delta
    def backward(self, output):
        return output * (1 - output)


# 全连接层类，记录除输入层外得每一层得信息，包括输入输出向量维数，权向量矩阵和偏置向量和误差向量
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        """
        构造函数
        :param input_size: 输入向量维数
        :param output_size: 输出向量维数，就等于该层的神经元数目
        :param activator: 一个SigmoidActivator对象
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.delta = np.zeros((output_size, 1))
        self.W_grad = np.zeros((output_size, input_size))
        self.b_grad = np.zeros(output_size)

        self.input = np.zeros((input_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        """
        前向计算该层得输出值
        :param input_array: 输入特征向量
        """
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, self.input) + self.b)

    def backward(self):
        """
        反向计算该层的误差\delta和w与b的梯度
        """
        # self.delta = self.activator.backward(self.output) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(self.delta, self.input.T)
        self.b_grad = self.delta

    def update(self, learning_rate):
        """
        更新该层的权向量矩阵和偏置向量
        :param learning_rate: 学习速率
        """
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


def label_analysis(label):
    j = 0
    max_one = 0
    for i in range(len(label)):
        if label[i] > max_one:
            j = i
            max_one = label[i]
    return j


# 神经网络类
class Network(object):
    def __init__(self, layers):
        """
        构造函数
        :param layers: 一个向量，代表了从输入至输出各层的维数，即神经元个数
        """
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator()))

    def predict(self, sample):
        """
        正向计算模型输出
        :param sample: 样本特征向量
        :return: 类别向量
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        训练函数
        :param labels: 训练集标签矩阵
        :param data_set: 训练集特征矩阵
        :param rate: 学习速率
        :param epoch: 迭代次数
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d].reshape(len(labels[d]), 1), data_set[d].reshape(len(data_set[0]), 1),
                                      rate)

    def train_one_sample(self, label, sample, rate):
        """
        用一个样本更新权向量矩阵和偏置向量
        :param label: 样本标签
        :param sample: 样本特征
        :param rate: 学习速率
        """
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        """
        计算梯度
        :param label: 样本实际标签
        :return:
        """
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        self.layers[-1].delta = delta
        self.layers[-1].backward()
        self.layers[-2].delta = self.layers[-2].activator.backward(self.layers[-2].output) * np.dot(self.layers[-1].W.T,
                                                                                                    self.layers[
                                                                                                        -1].delta)
        self.layers[-2].backward()

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def evaluate(self, data_test, label_test):
        error = 0
        total = len(data_test)
        for i in range(total):
            predict = self.predict(data_test[i].reshape(len(data_test[i]), 1))
            if label_analysis(predict) != label_analysis(label_test[i]):
                error += 1
        print('错误率：%.3f' % (error * 1.0 / len(label_test)))
        return error * 1.0 / len(label_test)
