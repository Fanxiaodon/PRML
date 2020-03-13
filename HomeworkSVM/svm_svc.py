import struct
import numpy as np
from sklearn.svm import SVC


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

    with open(image_path, 'rb') as im_path:
        magic, num, rows, cols = struct.unpack('>IIII', im_path.read(16))
        images = np.fromfile(im_path, dtype=np.uint8).reshape(len(labels), 784)

    return images[:number], labels[:number]


if __name__ == "__main__":
    images_train, labels_train = load_mnist(number=10000)
    images_test, labels_test = load_mnist(kind='t10k', number=1000)
    clf = SVC(C=500, kernel='rbf', gamma='scale')
    clf.fit(images_train, labels_train)
    score = clf.score(images_test, labels_test)
    print('得分：' + str(score))
    labels_result = clf.predict(images_test)
    print('实际标签：' + str(labels_test[0:100]))
    print('预测标签：' + str(labels_result[0:100]))
