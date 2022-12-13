import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('../data_loader')
sys.path.append('..')

import random
from PIL import Image
from torch.utils.data import DataLoader
from option import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size
from PIL import Image as pi


class Stage2_Dataset(data.Dataset):
    def __init__(self,train,size=crop_size,format='.png'):
        super(Stage2_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format

        '''
        hazepath: Your haze images
        clearpath: CLAHE handle your haze images
        (Data from PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors)
        clearlabels: collect clear images
        (You can collect any images by yourself)
        '''

        if self.train:
            self.hazepath = './train_images/RTTS/'
            self.clearpath = './train_images/CLAHE-RTTS/'
            self.clearlabel = './train_images/collect_images/'
            self.clearlabels = os.listdir(self.clearlabel)
            self.counts = len(self.clearlabels)
        else:
            self.hazepath = './train_images/RTTS/'
        self.haze_imgs = os.listdir(self.hazepath)


    def __getitem__(self, index):
        sname = self.haze_imgs[index]
        haze=Image.open(self.hazepath+sname)

        if not self.train:
            sizei = haze.size
            h, w = (sizei[0] // 4) * 4, (sizei[1] // 4) * 4

            haze = FF.crop(haze, 0, 0, w, h)
            if h > 2000 or w > 2000:
                haze = haze.resize((512,512),pi.ANTIALIAS)
            haze = tfs.ToTensor()(haze.convert('RGB'))
            return haze,sname

        if self.train:
            index_label = random.randint(0, self.counts-1)

            labelname = self.clearlabels[index_label]
            realclear = Image.open(self.clearlabel+labelname)

            gtname = sname.replace('.jpeg', '.png')
            clear = Image.open(self.clearpath + gtname)

            haze = haze.resize((self.size, self.size), pi.ANTIALIAS)
            clear = clear.resize((self.size, self.size), pi.ANTIALIAS)
            realclear = realclear.resize((self.size, self.size), pi.ANTIALIAS)

            haze = tfs.ToTensor()(haze.convert('RGB'))
            clear = tfs.ToTensor()(clear.convert('RGB'))
            realclear = tfs.ToTensor()(realclear.convert('RGB'))

            return haze,clear,realclear

    def __len__(self):
        return len(self.haze_imgs)

import os
pwd=os.getcwd()
print(pwd)


ITS_train_loader=DataLoader(dataset=Stage2_Dataset(train=True,size=crop_size),batch_size=BS,shuffle=True)
ITS_test_loader=DataLoader(dataset=Stage2_Dataset(train=False,size='whole img'),batch_size=1,shuffle=False)


if __name__ == "__main__":
    pass

