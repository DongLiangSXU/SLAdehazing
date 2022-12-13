import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('../data_loader')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size
from PIL import Image as pi


# 1_1_0.90179.png 1_1.png
class Stage1_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(Stage1_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        '''
        train : only need collect clear images
        test  : need paired images
        '''
        if not self.train:
            self.haze_imgs_dir=os.listdir(os.path.join(path,'lowq'))
            self.haze_imgs=[os.path.join(path,'lowq',img) for img in self.haze_imgs_dir]
            self.clear_dir=os.path.join(path,'clear')
        if self.train:
            self.rclearpath = path
            self.rclear_imgs_dir=os.listdir(self.rclearpath)
            self.rclear_imgs = [os.path.join(self.rclearpath, img) for img in self.haze_imgs_dir]


    def __getitem__(self, index):

        if not self.train:
            haze=Image.open(self.haze_imgs[index])
            img=self.haze_imgs[index]
            clear_name=img.split('/')[-1]
            flagnum = int(clear_name.split('.')[0])
            clear = Image.open(os.path.join(self.clear_dir, clear_name))
            h,w = clear.size
            newh,neww = (h//4)*4,(w//4)*4

            haze = FF.crop(haze, 0, 0, neww, newh)
            clear = FF.crop(clear, 0, 0, neww, newh)
            ohaze = tfs.ToTensor()(haze.convert('RGB'))
            haze = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(ohaze)
            clear = tfs.ToTensor()(clear.convert('RGB'))
            return haze, clear,ohaze,flagnum

        if self.train:
            rclear = Image.open(self.rclear_imgs[index])
            rclear = rclear.resize((self.size, self.size), pi.ANTIALIAS)
            rclear = tfs.ToTensor()(rclear.convert('RGB'))
            return rclear

    def __len__(self):
        if self.train:
            return len(self.rclear_imgs)
        else:
            return len(self.haze_imgs)

import os
pwd=os.getcwd()
print(pwd)

ITS_train_loader=DataLoader(dataset=Stage1_Dataset('./train_images/collect_images',train=True,size=crop_size),batch_size=BS,shuffle=True)
ITS_test_loader=DataLoader(dataset=Stage1_Dataset('/SOTS',train=False,size='whole img'),batch_size=1,shuffle=False)

if __name__ == "__main__":
    pass

