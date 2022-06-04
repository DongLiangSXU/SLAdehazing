import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Upproject(nn.Module):
   def __init__(self,in_channels,nf):
       super(Upproject,self).__init__()
       # self.upsample = F.upsample_bilinear
       self.upsample = F.interpolate
       self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=nf,stride=1,kernel_size=3,padding=1,bias=True)
       # self.relu = nn.LeakyReLU(0.2,inplace=True)
       self.conv2 = nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=3,stride=1,padding=1,bias=True)
       self.relu2 = nn.LeakyReLU(0.2)


   def forward(self, input, to_cat):
        shape_out = input.data.size()
        shape_out = shape_out[2:4]
        # print(shape_out)
        x1 = self.upsample(input,size=(shape_out[0]*2,shape_out[1]*2),mode='bilinear',align_corners=True)
        x1 = torch.cat([x1, to_cat], dim=1)
        # x1 = self.upsample(x1,size=(shape_out[0]*2,shape_out[1]*2))
        x2 = self.conv1(x1)
        # x2 = self.relu(x2)
        x3 = self.conv2(x2)
        x3 = self.relu2(x3)
        return x3

class DenseNet_pytorch(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(DenseNet_pytorch, self).__init__()
        # self.model = models.resnet34(pretrained=False)
        self.model = models.densenet169(pretrained=False)
        # self.model.load_state_dict(torch.load(Windows_filepath+'densenet169-b2777c0a.pth'))
        self.conv0 = self.model.features.conv0
        self.norm0 = self.model.features.norm0
        self.relu0 = self.model.features.relu0
        self.pool0 = self.model.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1 = self.model.features.denseblock1
        self.trans_block1 = self.model.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = self.model.features.denseblock2
        self.trans_block2 = self.model.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = self.model.features.denseblock3
        self.trans_block3 = self.model.features.transition3

        ############# Block4-down  16-16 ##############
        self.dense_block4 = self.model.features.denseblock4


        self.model_out = self.model.features.norm5
        self.model_relu = F.relu

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_out_channels = 1664
        self.midconv = nn.Conv2d(in_channels=self.model_out_channels,out_channels=self.model_out_channels,kernel_size=1,stride=1,padding=0,bias=True)
        # self.midrelu = nn.LeakyReLU(0.2,inplace=True)
        # 输出：1664
        self.up1 = Upproject(1920,832)
        self.up2 = Upproject(960,416)
        self.up3 = Upproject(480,208)
        self.up4 = Upproject(272, 104)
        self.finalconv = nn.Conv2d(in_channels=104,out_channels=1,kernel_size=3,stride=1,padding=1,bias=True)


    def forward(self, x):
        tempx = x
        shape_out = x.data.size()
        shape_out = shape_out[2:4]

        x0 = self.relu0(self.norm0(self.conv0(x)))
        tx1 =x0
        x0=self.pool0(x0)
        tx2 = x0
        x1 = self.trans_block1(self.dense_block1(x0))
        tx3 = x1
        x2 = self.trans_block2(self.dense_block2(x1))
        tx4 =x2

        x3 = self.trans_block3(self.dense_block3(x2))

        x4 = self.dense_block4(x3)
        finnalout = self.model_out(x4)
        finnalout = self.model_relu(finnalout)

        mid = self.midconv(finnalout)
        # output:640*8*8
        up1 = self.up1(mid, tx4)
        # output:256*16*16
        up2 = self.up2(up1, tx3)
        # output:128*40*40
        up3 = self.up3(up2, tx2)
        # output:64*80*80
        up4 = self.up4(up3, tx1)

        result = self.finalconv(up4)

        return result

def nyu_resize(img, resolution=512, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, resolution), preserve_range=True, mode='reflect', anti_aliasing=True )

def DepthNorm(x, maxDepth):
    return maxDepth / x


class NormDepth(nn.Module):
    def __init__(self,iskt=False,isfun=False):
        super(NormDepth, self).__init__()
        self.upsample = F.upsample_bilinear
        self.depth_es = DenseNet_pytorch(3, 3)
        self.isfun = isfun
        if iskt:
            self.depth_es.load_state_dict(torch.load('./pth_path/depth_pth/kitti.pth'))
        else:
            self.depth_es.load_state_dict(torch.load('./pth_path/depth_pth/my.pth'))

        self.threeshold = nn.Threshold(1, 1)

    def forward(self, input):

        depth = self.depth_es(input)

        shape = input.data.size()
        shape = shape[2:4]
        depth = self.upsample(depth, size=shape)
        depth = self.threeshold(depth)
        depth = 1 / depth

        return depth

