from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from  torchvision import datasets
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import random
import numpy as np
from torchtoolbox.transform import Cutout
random.seed(2022)

class Loader2(Dataset):
    def __init__(self,train=True):
        self.train = train

        if  self.train:
            self.data=pd.read_csv('train_data.csv').values
            
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomAffine(10),
                #transforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.5),  # 加入1
                Cutout(),
                transforms.ToTensor(),
                transforms.RandomErasing(p = 0.3,scale=(0.02, 0.33), ratio=(0.3, 3.3),value=(254/255,0,0)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]), ])
        else:
            self.data=pd.read_csv('val_data.csv').values
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                           ])


    def __getitem__(self,index):

        #pic_pth = os.path.join('./',self.data[index][0])
        pic=Image.open(self.data[index][0]).convert('RGB')
        label = self.data[index][1]
        pic = self.transforms(pic)
        return pic,label


    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_loader  = Loader2(train=True)
    val_loader  = Loader2(train=False)
    print(len(train_loader))
    print(len(val_loader))
    for img,label in val_loader:
        print(img)
        print(label)
        break




