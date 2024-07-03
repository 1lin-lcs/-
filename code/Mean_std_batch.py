import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

# 读取图片
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8), -1)
    return cv_img

# 加权标准差计算
def cal_sum_std(numArray,stdArray,size):
    TotalNum=np.sum(numArray)
    TotalStd=np.zeros(3)
    for i in range(0,3):
        for j in range(0,size):
            TotalStd[i]+=numArray[j]*np.square(stdArray[j][i])
    TotalStd=TotalStd/TotalNum
    return np.sqrt(TotalStd)

# img_h, img_w = 32, 32
each_mean=np.ndarray((20,3))
each_std=np.ndarray((20,3))
each_num=np.ndarray(20)
img_h, img_w = 224, 224
means, stdevs = [], []


TRAIN_DATASET_PATH = r"E:\文件\Documents\人工智能学习\CIFAR-10\DataSet"

image_fns=[]

path=os.listdir(TRAIN_DATASET_PATH)

# 文件夹下标
cur=0

# 按文件夹分批计算均值和标准差
for p in path:
    image_folder=os.path.join(TRAIN_DATASET_PATH,p)
    imgdir_list=os.listdir(image_folder)
    for i in imgdir_list:
        img_list = []
        image_fns=glob(os.path.join(image_folder,i)+"\\*.png")
        each_num[cur]=len(os.listdir(os.path.join(image_folder,i)))
        # 加载图片内容
        for single_img_path in tqdm(image_fns):
            try:    
                img = cv_imread(single_img_path)
                img = cv2.resize(img, (img_w, img_h))
                img = img[:, :, :, np.newaxis]
                img_list.append(img)
            except:
                continue    


        imgs = np.concatenate(img_list, axis=3)
        imgs = imgs.astype(np.float32) / 255.
        # 分通道计算均值和标准差
        for i in range(3):
            pixels = imgs[:, :, i, :].ravel()  # 拉成一行
            each_mean[cur,i]=np.mean(pixels)
            each_std[cur,i]=np.std(pixels)
            
        # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
        # each_mean[cur].reverse()
        # each_std[cur].reverse()
        cur+=1
        
'''
normMean = [[0.58885306 0.56035167 0.52537978]
 [0.445905   0.45338732 0.46990094]
 [0.42395672 0.49188221 0.49007836]
 [0.41560435 0.45679638 0.49580261]
 [0.37903085 0.46565044 0.47165945]
 [0.41636834 0.4638246  0.49951124]
 [0.34470418 0.43814701 0.4701139 ]
 [0.41680703 0.48006269 0.50218612]
 [0.55502445 0.52544218 0.49015144]
 [0.47827563 0.48561606 0.49867135]
 [0.5896256  0.56042296 0.52858359]
 [0.4576692  0.4637816  0.48157734]
 [0.42505467 0.48772034 0.48160753]
 [0.41475859 0.45270967 0.49222919]
 [0.37010926 0.46075016 0.47062176]
 [0.41814548 0.47196692 0.50359988]
 [0.34983319 0.44034043 0.46907705]
 [0.41746467 0.47807851 0.49988434]
 [0.55162024 0.52503455 0.4910349 ]
 [0.47589591 0.48250952 0.49837208]]
normStd = [[0.26580182 0.24058896 0.2505303 ]
 [0.27422887 0.26509902 0.26731333]
 [0.24311076 0.22097783 0.22702749]
 [0.25811893 0.2524308  0.25730368]
 [0.21136242 0.20639792 0.21716306]
 [0.24903703 0.24393801 0.25078338]
 [0.21995185 0.21822083 0.22860816]
 [0.2517744  0.2437975  0.24285476]
 [0.25099528 0.2401825  0.24892905]
 [0.28066334 0.26874968 0.2677452 ]
 [0.26625004 0.241716   0.24602275]
 [0.27816272 0.26903927 0.27150583]
 [0.24489783 0.21966177 0.22620381]
 [0.25570577 0.24953119 0.25131446]
 [0.21435256 0.20618658 0.21750185]
 [0.24800307 0.24174874 0.24698544]
 [0.22248277 0.21939    0.22941114]
 [0.24949761 0.24388896 0.24318324]
 [0.2545394  0.24375105 0.25441074]
 [0.2813167  0.26932731 0.26794869]]
 normMean = [0.4915021419525146, 0.48222376108169557, 0.4467353209853172]
 normStd = [0.24617893723828543, 0.24100179205311906, 0.25184986282754784]

 normMean = [0.4915021419525146, 0.48222376108169557, 0.4467353209853172]
 normStd = [0.24633002172920254, 0.24083635024354832, 0.25146039723774327]
'''



# 计算总的均值和标准差
total_mean = np.mean(each_mean, axis=0)
total_std = cal_sum_std(each_num,each_std,20)
#total_std = np.sqrt(np.mean(np.square(subset_stds), axis=0))

means=total_mean.tolist()
stdevs=total_std.tolist()

# 通道数据转换，OpenCV通道表示不一样
means.reverse()
stdevs.reverse()

print("normMean = {}".format(each_mean))
print("normStd = {}".format(each_std))
print("--------------------------------")
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))