import torch.nn.functional as F
from torch import nn
import torch

# 用于较浅的网络
class ResidualBlock(nn.Module):
    expansion = 1  # 扩展系数expansion表示的是单元输出与输入张量的通道数之比

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


# 用于较深的网络
class Bottleneck(nn.Module):
    expansion = 4  # 扩展系数expansion表示的是单元输出与输入张量的通道数之比

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 第一个卷积层，1x1，用于降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 第三个卷积层，1x1，用于升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        # 残差连接
        self.shortcut = nn.Sequential()
        # self.self_attention=SelfAttention(32,in_channels*self.expansion,in_channels*self.expansion,0.3)
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        # print(out.size())
        # self.self_attention(out)
        return out



class ResNetRaw(nn.Module):
    def __init__(self, block, num_blocks, classnum):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.__make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.__make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.__make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.__make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, classnum)
        # self.sf=nn.Softmax()

    def __make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 按照网页上的自适应平均池化层
        out = self.avgpool(out)
        out = out.reshape(x.size(0), -1)
        # out=F.avg_pool2d(out, 7)   #根据实际输入大小改
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNetRaw18(classNum):
    return ResNetRaw(ResidualBlock, [2, 2, 2, 2], classNum)


def ResNetRaw34(classNum):
    return ResNetRaw(ResidualBlock, [3, 4, 6, 3], classNum)


def ResNetRaw50(classNum):
    return ResNetRaw(Bottleneck, [3, 4, 6, 3], classNum)


def ResNetRaw101(classNum):
    return ResNetRaw(Bottleneck, [3, 4, 23, 3], classNum)


def ResNetRaw152(classNum):
    return ResNetRaw(Bottleneck, [3, 8, 36, 3], classNum)


if __name__ == "__main__":
    # net=ResNet(ResidualBlock,[3, 4, 6, 3],54).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNetRaw50(10).to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    y = net(x)
    print(y.size())
    print(y)
    print(net)