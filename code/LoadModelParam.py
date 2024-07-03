import torch
import os

# 用于加载模型
class LoadModelParam():
    def __init__(self,
                 path:str=None,
                 model:torch.nn.Module=None,
                 strict:bool=True):
        # 模型文件路径
        self.modelPath=None
        # 模型类
        self.model=None
        # 是否要求模型文件中的模块与类中的模块对应
        self.strict=strict
        if path!=None and os.path.exists(path) and os.path.isfile(path):
            self.modelPath=path
            if model!=None:
                self.model=self.LoadParaState(self.modelPath,model,self.strict)
                self.SetParam()
            
        
    def SetModelPath(self,path:str):
        self.modelPath=path
        
    def GetModelPath(self)->str:
        return self.modelPath
    
    def SetParam(self,
                 path:str=None,
                 model:torch.nn.Module=None,
                 strict:bool=False):
        modelPath=None
        m_model=None
        if path==None:
            modelPath=self.modelPath
        else:
            modelPath=path
        if model==None:
            m_model=self.model
        else:
            m_model=model
        self.model=self.LoadParaState(modelPath,m_model,strict)
        
    def GetModel(self)->torch.nn.Module:
        return self.model
        
    @staticmethod
    def LoadParam(path:str):
        if os.path.exists(path) and os.path.isfile(path):
            return torch.load(path)
        else:
            return None

    # 加载模型
    @staticmethod
    def LoadParaState(path:str,model:torch.nn.Module,strict:bool=False):
        m=model
        if os.path.exists(path) and os.path.isfile(path):
            m.load_state_dict(torch.load(path),strict=strict)
            return m
        else:
            return None

    # 将save()保存的数据转换成用state_dict()保存的数据
    @staticmethod
    def Param2StateDict(filePath:str,savePath:str):
        if os.path.exists(filePath) and os.path.isfile(filePath):
            _model=torch.load(filePath)
            torch.save(_model.state_dict(),savePath)
            
        
if __name__=='__main__':
    from ResNet import *
    from torchsummary import summary
    modelPath=r"E:\文件\Documents\人工智能学习\CIFAR-10\ACC_RestNetState_dict_809.pth"
    model=LoadModelParam(modelPath,ResNet50(10),False).GetModel()
    print(summary(model.to(device),(3,224,224)))




