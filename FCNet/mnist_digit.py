import FullConnect
import numpy as np
import struct
import time


def load_mnist(kind='train', number=5000):
    """
    加载mnist手写数据集
    :param number: 取前number个数据
    :param kind: 文件类型
    :return: 训练集的特征矩阵和labels
    """
    labels_path = kind + '-labels.idx1-ubyte'
    image_path = kind + '-images.idx3-ubyte'

    with open(labels_path, 'rb') as lb_path:
        magic, n = struct.unpack('>II', lb_path.read(8))
        labels = np.fromfile(lb_path, dtype=np.uint8)
    labels_mat = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        labels_mat[i] = norm(labels[i])

    with open(image_path, 'rb') as im_path:
        magic, num, rows, cols = struct.unpack('>IIII', im_path.read(16))
        images = np.fromfile(im_path, dtype=np.uint8).reshape(len(labels), 784)

    return images[:number], labels_mat


def norm(label):
    """
    将label转换为一个10维向量，其中只有一个值为0.9，其它为0.1
    :param label: 标签
    """
    label_vector = []
    for i in range(10):
        if label == i:
            label_vector.append(0.9)
        else:
            label_vector.append(0.1)
    return label_vector


if __name__ == "__main__":
    error = 1
    images_train, labels_train = load_mnist(number=60000)
    images_test, labels_test = load_mnist(kind='t10k', number=10000)

    net = FullConnect.Network([784, 300, 10])
    while True:
        ticks = time.time()
        net.train(labels_train[:10000], images_train[:10000], 0.005, 1)
        error1 = net.evaluate(data_test=images_test[:2000], label_test=labels_test[:2000])
        print("训练用时：%.2f s" % (time.time() - ticks))
        if error1 > error:
            print('训练结束')
            break
        error = error1
