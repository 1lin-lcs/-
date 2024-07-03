import torch
import tqdm
from ResNet import *
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import csv
from LoadModelParam import *
from ResNetRGA import *
from ResNetAtt import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

normMean = [0.4915021419525146, 0.48222376108169557, 0.4467353209853172]
normStd = [0.24633002172920254, 0.24083635024354832, 0.25146039723774327]

model=LoadModelParam("../model/ResNetRGA.pth",ResNet50(10),True).GetModel()
model=model.to(device)

transform=transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(normMean,normStd)
        ]
    )

path=r"/hy-tmp/test"
files=os.listdir(path)

#{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

dic=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
csvFile=r"/hy-tmp/answer.csv"
f=open(csvFile,"w")
writer=csv.writer(f)
writer.writerow(["id","label"])
# print(type(model))

model.eval()
for f in tqdm(files):
    filepath=os.path.join(path,f)
    image=Image.open(filepath)
    image=transform(image)
    image=image.unsqueeze(0)
    with torch.no_grad():
        image=image.to(device)
        output=model(image)
        pre_lab=torch.argmax(output,dim=1)
        result=pre_lab.item()
        writer.writerow([os.path.basename(filepath).split('.')[0],dic[result]])