import torch
from torch import nn

from RGA import RGAModule
from ResNet import Bottleneck
from thop import profile


class Resnet50_RGA(nn.Module):
    def __init__(self, class_num, block=Bottleneck, layers=None, height=56, width=56, spa_on=True, cha_on=True, s_ratio=8, c_ratio=8, d_ratio=7):
        super(Resnet50_RGA, self).__init__()
        if layers is None:
            layers = [3, 4, 6, 3]
        self.in_channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.__make_layer(block, 64, layers[0],stride=1)
        self.layer2 = self.__make_layer(block,128,layers[1], stride=2)
        self.layer3 = self.__make_layer(block,256,layers[2], stride=2)
        self.layer4 = self.__make_layer(block,512,layers[3],stride=2)
        # RGA注意力机制
        self.rga1=RGAModule(256,height*width,use_spatial=spa_on,use_channel=cha_on,cha_ratio=c_ratio,spa_ratio=s_ratio,down_ration=d_ratio)
        self.rga2=RGAModule(512,(height//2)*(width//2),use_spatial=spa_on,use_channel=cha_on,cha_ratio=c_ratio,spa_ratio=s_ratio,down_ration=d_ratio)
        self.rga3=RGAModule(1024,(height//4)*(width//4),use_spatial=spa_on,use_channel=cha_on,cha_ratio=c_ratio,spa_ratio=s_ratio,down_ration=d_ratio)
        self.rga4=RGAModule(2048,(height//8)*(width//8),use_spatial=spa_on,use_channel=cha_on,cha_ratio=c_ratio,spa_ratio=s_ratio,down_ration=d_ratio)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,class_num)

    def __make_layer(self,block,out_channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for s in strides:
            layers.append(block(self.in_channels,out_channels,s))
            self.in_channels=out_channels*block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.rga1(x)
        x=self.layer2(x)
        x=self.rga2(x)
        x=self.layer3(x)
        x=self.rga3(x)
        x=self.layer4(x)
        x=self.rga4(x)
        # 按照网页上的自适应平均池化层
        x=self.avgpool(x)
        x=x.reshape(x.size(0),-1)
        # x=F.avg_pool2d(x, 7)   #根据实际输入大小改
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x

if __name__=="__main__":
    #net=ResNet(ResidualBlock,[3, 4, 6, 3],54).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=Resnet50_RGA(10).to(device)
    x=torch.randn(1,3,224,224).to(device)
    # y=net(x)
    # print(y.size())
    # print(y)
    # print(net)
    flops, params = profile(net, inputs=(x,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')