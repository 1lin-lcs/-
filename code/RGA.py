from torch import nn
import torch

#https://www.bilibili.com/read/cv16615641/#circle=on

class RGAModule(nn.Module):
    def __init__(self,in_channels,in_spatial,use_spatial=True,use_channel=True,cha_ratio=8,spa_ratio=8,down_ration=7):
        super(RGAModule, self).__init__()
        self.in_channel=in_channels
        self.in_spatial=in_spatial
        self.inter_channel=in_channels//cha_ratio
        self.inter_spatial=in_spatial//spa_ratio
        # 是否使用空间注意力
        self.use_spatial=use_spatial
        # 是否使用通道注意力
        self.use_channel=use_channel
        if self.use_spatial:

            self.theta_spatial = nn.Sequential(nn.Conv2d(in_channels=self.in_channel,out_channels=self.inter_channel,kernel_size=1,stride=1,padding=0,bias=False),
                                               nn.BatchNorm2d(self.inter_channel),
                                               nn.ReLU())

            self.phi_spatial = nn.Sequential(nn.Conv2d(in_channels=self.in_channel,out_channels=self.inter_channel,kernel_size=1,stride=1,padding=0,bias=False),
                                             nn.BatchNorm2d(self.inter_channel),
                                             nn.ReLU())

            self.gg_spatial = nn.Sequential(nn.Conv2d(in_channels=self.in_spatial*2,out_channels=self.inter_spatial,kernel_size=1,stride=1,padding=0,bias=False),
                                            nn.BatchNorm2d(self.inter_spatial),
                                            nn.ReLU())

            self.gx_spatial = nn.Sequential(nn.Conv2d(in_channels=self.in_channel,out_channels=self.inter_spatial,kernel_size=1,stride=1,padding=0,bias=False),
                                            nn.BatchNorm2d(self.inter_spatial),
                                            nn.ReLU())

            num_channel_s=1+self.inter_spatial
            self.W_spatial = nn.Sequential(nn.Conv2d(in_channels=num_channel_s,out_channels=num_channel_s//down_ration,kernel_size=1,stride=1,padding=0,bias=False),
                                           nn.BatchNorm2d(num_channel_s//down_ration),
                                           nn.ReLU(),
                                           nn.Conv2d(in_channels=num_channel_s//down_ration,out_channels=1,kernel_size=1,stride=1,padding=0,bias=False),
                                           nn.BatchNorm2d(1))

        if use_channel:

            self.theta_channel = nn.Sequential(nn.Conv2d(in_channels=self.in_spatial,out_channels=self.inter_spatial,kernel_size=1,stride=1,padding=0,bias=False),
                                             nn.BatchNorm2d(self.inter_spatial),
                                             nn.ReLU())

            self.phi_channel = nn.Sequential(nn.Conv2d(in_channels=self.in_spatial,out_channels=self.inter_spatial,kernel_size=1,stride=1,padding=0,bias=False),
                                             nn.BatchNorm2d(self.inter_spatial),
                                             nn.ReLU())

            self.gg_channel = nn.Sequential(nn.Conv2d(in_channels=self.in_channel*2,out_channels=self.inter_channel,kernel_size=1,stride=1,padding=0,bias=False),
                                            nn.BatchNorm2d(self.inter_channel),
                                            nn.ReLU())

            self.gx_channel = nn.Sequential(nn.Conv2d(in_channels=self.in_spatial,out_channels=self.inter_spatial,kernel_size=1,stride=1,padding=0,bias=False),
                                            nn.BatchNorm2d(self.inter_spatial),
                                            nn.ReLU())

            num_channel_c=1+self.inter_channel
            self.W_channel = nn.Sequential(nn.Conv2d(in_channels=num_channel_c,out_channels=num_channel_c//down_ration,kernel_size=1,stride=1,padding=0,bias=False),
                                           nn.BatchNorm2d(num_channel_c//down_ration),
                                           nn.ReLU(),
                                           nn.Conv2d(in_channels=num_channel_c//down_ration,out_channels=1,kernel_size=1,stride=1,padding=0,bias=False),
                                           nn.BatchNorm2d(1))
    def forward(self,x):
        b,c,h,w = x.size()
        if self.use_spatial:
            # 计算空间注意力的theta映射
            theta_xs=self.theta_spatial(x)
            # 计算空间注意力的phi映射
            phi_xs=self.phi_spatial(x)
            # 将theta映射转换为适合矩阵乘法的形式
            theta_xs=theta_xs.view(b,self.inter_channel,-1)
            # 调整theta映射的顺序，以便于与phi映射进行矩阵乘法
            theta_xs=theta_xs.permute(0,2,1)
            # 同样将phi映射转换为适合矩阵乘法的形式
            phi_xs=phi_xs.view(b,self.inter_channel,-1)
            # 计算空间注意力的Gs映射
            Gs=torch.matmul(theta_xs,phi_xs)
            # 将Gs映射转换回原始输入的空间形式
            Gs_in=Gs.permute(0,2,1).view(b,h*w,h,w)
            # 同样将Gs映射转换回原始输入的空间形式
            Gs_out=Gs.view(b,h*w,h,w)
            # 将Gs_in和Gs_out拼接在一起
            Gs_joint=torch.cat((Gs_in,Gs_out),dim=1)
            # 应用全局聚合网络gg_spatial
            Gs_joint=self.gg_spatial(Gs_joint)
            # 计算空间注意力的位置特征g_xs
            g_xs=self.gx_spatial(x)
            # 计算g_xs的平均值
            g_xs=torch.mean(g_xs,dim=1,keepdim=True)
            # 将g_xs和Gs_joint拼接在一起
            ys=torch.cat((g_xs,Gs_joint),dim=1)
            # 应用全局加权网络W_spatial
            w_ys=self.W_spatial(ys)
            if not self.use_channel:
                # 如果只使用空间注意力，则直接将注意力权重与输入x相乘
                out=torch.sigmoid(w_ys.expand_as(x))*x
                return out
            else:
                # 如果同时使用空间和通道注意力，则先将空间注意力权重与输入x相乘，然后再进行通道注意力计算
                x=torch.sigmoid(w_ys.expand_as(x))*x
        # 通道注意力部分
        if self.use_channel:
            # 转换输入x为适合计算通道注意力的形式
            xc=x.view(b,c,-1).permute(0,2,1).unsqueeze(-1)
            # 计算通道注意力的theta映射
            theta_xc=self.theta_channel(xc).squeeze(-1).permute(0,2,1)
            # 计算通道注意力的phi映射
            phi_xc=self.phi_channel(xc).squeeze(-1)
            # 计算通道注意力的Gs映射
            Gc=torch.matmul(theta_xc,phi_xc)
            # 将Gs映射转换回原始输入的通道形式
            Gc_in=Gc.permute(0,2,1).unsqueeze(-1)
            # 同样将Gs映射转换回原始输入的通道形式
            Gc_out=Gc.unsqueeze(-1)
            # 将Gc_in和Gc_out拼接在一起
            Gc_joint=torch.cat((Gc_in,Gc_out),1)
            # 应用全局聚合网络gg_channel
            Gc_joint=self.gg_channel(Gc_joint)
            # 计算通道注意力的位置特征g_xc
            g_xc=self.gx_channel(xc)
            # 计算g_xc的平均值
            g_xc=torch.mean(g_xc,dim=1,keepdim=True)
            # 将g_xc和Gc_joint拼接在一起
            yc=torch.cat((g_xc,Gc_joint),dim=1)
            # 应用全局加权网络W_channel并转置
            w_yc=self.W_channel(yc).transpose(1,2)
            # 将注意力权重与输入x相乘
            out=torch.sigmoid(w_yc)*x
            return out
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=RGAModule(2048,7*7,use_spatial=True,use_channel=True).to(device)
    x=torch.randn((1,2048,7,7)).to(device)
    y=model(x)
    print(y.shape)