from ResNet import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms,datasets
from LoadModelParam import *
import os
import torchvision
from ResNetAtt import *
from ResNetRGA import Resnet50_RGA

# 计算得到的均值和标准差

normMean = [0.4915021419525146, 0.48222376108169557, 0.4467353209853172]
normStd = [0.24633002172920254, 0.24083635024354832, 0.25146039723774327]

if not os.path.exists("./model"):
    os.mkdir("./model")

# 预处理
data_transforms={'train':transforms.Compose(
                    [
                        transforms.RandomRotation(45),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        # transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),
                        # transforms.RandomGrayscale(p=0.025),
                        transforms.ToTensor(),
                        transforms.Normalize(normMean,normStd)
                        ]
                    ),
                 'valid':transforms.Compose(
                     [
                         transforms.RandomRotation(45),
						 transforms.RandomHorizontalFlip(p=0.5),
                         transforms.RandomVerticalFlip(p=0.5),
                         # transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),
                         # transforms.RandomGrayscale(p=0.025),
                         transforms.ToTensor(),
                         transforms.Normalize(normMean,normStd)
                         ]
                    )
}



batch_size=32
# 记录训练过程
f=open("./output.txt","w")
data_dir=r"/hy-tmp/DataSet"
train_dir=data_dir+"/train"
validation_dir=data_dir+"/valid"
# 加载数据集
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
dataloaders={x:DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True) for x in ['train','valid']}
dataset_sizes={x:len(image_datasets[x]) for x in ['train','valid']}

#print(image_datasets['train'].class_to_idx)

#训练的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据集的长度
train_data_size=dataset_sizes['train']
validation_data_size=dataset_sizes['valid']
# 初始化模型
test=Resnet50_RGA(10)
#test=LoadModelParam(r"/hy-tmp/HModel/ACC_ResNet_state_dict_703.pth",ResNet50(10),False).GetModel()
test=test.to(device)    #用GPU，这个可以不用赋值
#损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)      #用GPU，这个可以不同赋值
#优化器
learning_rate=0.01
optim=torch.optim.SGD(test.parameters(),learning_rate,momentum=0.9)

#记录训练的次数
total_train_step=0
#测试的次数
total_test_step=0
#训练的轮数
epoch=50

writer=SummaryWriter("./log")

for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i+1))
    f.write("----------第{}轮训练开始----------\n".format(i+1))
    #训练步骤
    #如果有Dropout Batchnorm一定要要调用xxx.train()[train] 和xxx.eval()[test]
    test.train()
    total_train_loss=0
    for data in dataloaders['train']:
        optim.zero_grad()
        imgs,targets=data
        imgs=imgs.to(device)    #用GPU
        targets=targets.to(device)
        output=test(imgs)
        loss=loss_fn(output,targets)
        loss.backward()
        optim.step()
        total_train_step+=1
        total_train_loss+=loss.item()# 直接加爆内存
        if total_train_step%100==0:
            print("训练次数{}，Loss:{}".format(total_train_step,loss.item()))
            f.write("训练次数{}，Loss:{}\n".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    print("整体训练集上的Loss:{}".format(total_train_loss))
    #测试步骤开始
    total_test_loss=0
    total_accuray=0
    test.eval()
    with torch.no_grad():
        for data in dataloaders['valid']:
        #for data in dataloaders['train']:
            imgs,targets=data
            imgs=imgs.to(device)    #用GPU
            targets=targets.to(device)
            output=test(imgs)
            loss=loss_fn(output,targets)
            total_test_loss+=loss.item()
            accuray=(output.argmax(1)==targets).sum()
            total_accuray+=accuray
    print("整体测试集上的Loss:{}".format(total_test_loss))
    f.write("整体测试集上的Loss:{}\n".format(total_test_loss))
    print("整体测试集上的Accuray:{}".format(total_accuray/validation_data_size))       #这里要改总数
    f.write("整体测试集上的Accuray:{}\n".format(total_accuray/validation_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("accuray",total_accuray/validation_data_size,total_test_step)     #这里要改总数
    total_test_step+=1

    torch.save(test.state_dict(),"./model/test_{}_ACC_{}.pth".format(i,total_accuray/validation_data_size))
    print("模型已保存")
    f.write("模型已保存")
writer.close()
f.close()