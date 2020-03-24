import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, 3)  # 卷积核为方阵的时候可以只传入一个参数
        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_net(self, trainloader, device):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零
            self.optimizer.zero_grad()

            # 前向传播+计算损失函数+反向传播
            output = self(inputs)
            loss = self.criterion(output, labels)
            loss.backward()
            # 更新权系数
            self.optimizer.step()

            # 每2000个数据打印一次平均损失
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 1000 mini-batches
                print('%5d loss: %.3f' %
                      (i + 1, running_loss / 1000))
                running_loss = 0.0

    def cal_accuracy(self, testloader, device):
        correct = 0
        total = 0
        # 不进行autograd
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on test images: %d %%' % (
                100 * correct / total))
        return correct / total


if __name__ == '__main__':
    # 优先选择gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=False, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         transform=transform, download=False)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=16, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=16, shuffle=True, num_workers=4)

    net = CnnNet()
    net.to(device)
    accuracy = 0
    for epoch in range(20):
        print(epoch + 1, end=':')
        net.train_net(trainloader, device)
        acc = net.cal_accuracy(testloader, device)
        if acc < accuracy:
            break
        accuracy = acc
    print("Finished training")
