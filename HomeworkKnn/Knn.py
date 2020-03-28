import MnistLoader
from sklearn.neighbors import KNeighborsClassifier as KNN
import time

if __name__ == '__main__':
    tick = time.time()
    trainloader = MnistLoader.MyDataSet('data', train=True)
    testloader = MnistLoader.MyDataSet('data', train=False)

    train_img_set = []
    train_label_set = []
    test_img_set = []
    test_label_set = []

    for img, label in trainloader:
        img = img.reshape(1, 784)
        train_img_set.append(img[0, :])
        train_label_set.append(label)
    # for img, label in testloader:
    #     img = img.reshape(1, 784)
    #     test_img_set.append(img[0, :])
    #     test_label_set.append(label)

    kn = KNN(n_neighbors=5, algorithm='auto')
    kn.fit(train_img_set[:10000], train_label_set[:10000])
    error = 0
    for img, label in testloader:
        img = img.reshape(1, 784)
        pre = kn.predict(img)
        if pre != label:
            print('分类错误：预测结果：%d' % pre, end=';')
            print('真实结果：%d' % label)
            error += 1
    print('分类正确率：%.2f' % (1 - error/10000.0))
    print('完成10000数据分类，分类用时：%d s' % (time.time() - tick))
