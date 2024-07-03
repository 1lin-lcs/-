import torch
import torch.nn as nn

# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self,in_planes,ratio=16):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveAvgPool2d(1)
        # 第一个卷积层，将输入通道数减少到1/ratio
        self.fc1=nn.Conv2d(in_planes,in_planes//ratio,1,bias=False)
        self.relu1=nn.ReLU()
        # 第二个卷积层，将通道数还原
        self.fc2=nn.Conv2d(in_planes//ratio,in_planes,1,bias=False)
        
        self.sigmod=nn.Sigmoid()
        
    def forward(self,x):
        avg_out=self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out=self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out=avg_out+max_out
        return out

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        # 确认kernel_size为3或7
        assert kernel_size in (3,7),'kernel size must be 3 or 7'
        padding=3 if kernel_size==7 else 1
        self.conv1=nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmod=nn.Sigmoid()
        
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        # 在通道维度上拼接平均值和最大值
        x=torch.cat([avg_out,max_out],dim=1)
        # 卷积特征提取
        x=self.conv1(x)
        return self.sigmod(x)