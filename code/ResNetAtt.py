import torch.nn.functional as F

from Attention import *
# from self_attention import *
from thop import profile

#用于较浅的网络
class ResidualBlock(nn.Module):
    expansion=1 #扩展系数expansion表示的是单元输出与输入张量的通道数之比
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        # 残差连接
        self.shortcut=nn.Sequential()
        if stride!=1 or in_channels!=out_channels*self.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels*self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )

    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        # 添加残差连接
        out=out+self.shortcut(x)
        out=F.relu(out)
        return out
    
#用于较深的网络
class Bottleneck(nn.Module):
    expansion=4 #扩展系数expansion表示的是单元输出与输入张量的通道数之比
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        # 第一个卷积层，1x1，用于降维
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        # 第三个卷积层，1x1，用于升维
        self.conv3=nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
        # 残差连接
        self.shortcut=nn.Sequential()
        if stride!=1 or in_channels!=out_channels*self.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels*self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
    
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        out=out+self.shortcut(x)
        out=F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,num_blocks,classnum):
        super().__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        # https://zhuanlan.zhihu.com/p/99261200#circle=on
        # 这里添加了注意力机制
        self.ca=ChannelAttention(self.in_channels)
        self.sa=SpatialAttention()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=self.__make_layer(block,64,num_blocks[0],stride=1)
        self.layer2=self.__make_layer(block,128,num_blocks[1],stride=2)
        self.layer3=self.__make_layer(block,256,num_blocks[2],stride=2)
        # 这里添加了注意力机制
        self.caL3=ChannelAttention(self.in_channels)
        self.saL3=SpatialAttention()
        self.layer4=self.__make_layer(block,512,num_blocks[3],stride=2)
        # 这里添加了注意力机制
        self.ca1=ChannelAttention(self.in_channels)
        self.sa1=SpatialAttention()
        # 网页这里有自适应平均池化?
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,classnum)

    # 根据输入生成ResidualBlock块或Bottleneck块
    def __make_layer(self,block,out_channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for s in strides:
            layers.append(block(self.in_channels,out_channels,s))
            self.in_channels=out_channels*block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        # 这里添加了注意力机制
        out=self.ca(out)*out
        out=self.sa(out)*out
        out=self.maxpool(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        # >_< 试试
        out=self.caL3(out)*out
        out=self.saL3(out)*out
        out=self.layer4(out)
        # 这里添加了注意力机制
        out=self.ca1(out)*out
        out=self.sa1(out)*out
        # 按照网页上的自适应平均池化层
        out=self.avgpool(out)
        out=out.reshape(x.size(0),-1)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out
    
def ResNet18(classNum):
    return ResNet(ResidualBlock,[2, 2, 2, 2],classNum)


def ResNet34(classNum):
    return ResNet(ResidualBlock,[3, 4, 6, 3],classNum)


def ResNet50(classNum):
    return ResNet(Bottleneck,[3, 4, 6, 3],classNum)


def ResNet101(classNum):
    return ResNet(Bottleneck,[3, 4, 23, 3],classNum)


def ResNet152(classNum):
    return ResNet(Bottleneck,[3, 8, 36, 3],classNum)

if __name__=="__main__":
    #net=ResNet(ResidualBlock,[3, 4, 6, 3],54).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=ResNet50(10).to(device)
    x=torch.randn(1,3,224,224).to(device)
    # y=net(x)
    # print(y.size())
    # print(y)
    # print(net)
    flops, params = profile(net, inputs=(x,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')