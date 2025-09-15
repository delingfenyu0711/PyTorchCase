import torch  # 导入PyTorch深度学习框架
from torch.utils.data import DataLoader  # 用于批量加载数据
from torchvision import transforms  # 用于图像数据转换
from torchvision.datasets import MNIST  # 导入MNIST数据集
import matplotlib.pyplot as plt  # 用于可视化图像和结果


class Net(torch.nn.Module):
    """定义神经网络模型类，继承自PyTorch的Module基类"""

    def __init__(self):
        """初始化神经网络各层"""
        super().__init__()  # 调用父类的构造函数
        # 定义全连接层：输入为28*28的图像像素（784维），输出为64维
        self.fc1 = torch.nn.Linear(28*28, 64)
        # 第二个全连接层：输入64维，输出64维
        self.fc2 = torch.nn.Linear(64, 64)
        # 第三个全连接层：输入64维，输出64维
        self.fc3 = torch.nn.Linear(64, 64)
        # 输出层：输入64维，输出10维（对应0-9共10个数字）
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        """定义前向传播过程"""
        # 第一层全连接+ReLU激活函数
        x = torch.nn.functional.relu(self.fc1(x))
        # 第二层全连接+ReLU激活函数
        x = torch.nn.functional.relu(self.fc2(x))
        # 第三层全连接+ReLU激活函数
        x = torch.nn.functional.relu(self.fc3(x))
        # 输出层+log_softmax激活（适用于多分类任务）
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    """
    创建数据加载器
    参数is_train: 布尔值，True表示加载训练集，False表示加载测试集
    返回: 处理后的DataLoader对象
    """
    # 数据转换：将图像转为PyTorch张量（0-1范围）
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 加载MNIST数据集，若本地没有则自动下载
    data_set = MNIST(
        root="",  # 数据集保存路径（当前目录）
        train=is_train,  # 是否为训练集
        transform=to_tensor,  # 应用数据转换
        download=True  # 自动下载数据集
    )
    # 创建DataLoader，批量大小为15，打乱数据顺序
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    """
    评估模型在测试集上的准确率
    参数test_data: 测试数据集加载器
    参数net: 训练好的神经网络模型
    返回: 准确率（正确预测数/总样本数）
    """
    n_correct = 0  # 正确预测的样本数量
    n_total = 0    # 总样本数量
    
    # 关闭梯度计算（评估阶段不需要反向传播）
    with torch.no_grad():
        # 遍历测试集中的每个批次
        for (x, y) in test_data:
            # 前向传播：将图像展平为784维向量输入网络
            outputs = net.forward(x.view(-1, 28*28))
            # 遍历批次中的每个样本
            for i, output in enumerate(outputs):
                # 取输出最大值对应的索引作为预测结果，与真实标签比较
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    
    return n_correct / n_total  # 返回准确率


def main():
    """主函数：执行模型训练和评估的完整流程"""
    # 获取训练集和测试集的数据加载器
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    
    # 初始化神经网络模型
    net = Net()
    
    # 打印初始模型（随机权重）的准确率（接近10%，随机猜测水平）
    print("initial accuracy:", evaluate(test_data, net))
    
    # 定义优化器：使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 训练模型：迭代2个epoch（完整遍历训练集2次）
    for epoch in range(2):
        # 遍历训练集中的每个批次
        for (x, y) in train_data:
            # 清零梯度（防止梯度累积）
            net.zero_grad()
            # 前向传播：计算模型输出
            output = net.forward(x.view(-1, 28*28))
            # 计算损失：使用负对数似然损失（配合log_softmax使用）
            loss = torch.nn.functional.nll_loss(output, y)
            # 反向传播：计算梯度
            loss.backward()
            # 更新参数：根据梯度调整网络权重
            optimizer.step()
        
        # 每个epoch结束后，评估模型在测试集上的准确率
        print(f"epoch {epoch}, accuracy: {evaluate(test_data, net)}")
    
    # 可视化4个测试样本及其预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:  # 只展示前4个样本
            break
        # 对批次中第一个样本进行预测
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        # 绘制图像
        plt.figure(n)
        plt.imshow(x[0].view(28, 28), cmap="gray")  # 转为28x28灰度图
        plt.title(f"prediction: {int(predict)}")  # 显示预测结果
    plt.show()  # 显示所有图像


# 当脚本直接运行时，执行主函数
if __name__ == "__main__":
    main()