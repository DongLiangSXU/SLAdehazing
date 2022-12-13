import sys

sys.path.append("..")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from network import *
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from torchvision.transforms import functional as FF
import torchvision.utils as vutils
from PIL import Image as pi

warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir


print('log_dir :', log_dir)
print('model_name:', model_name)


models_ = {
    'trannet': BSnet(32, 1, istran=True),
    'atonet': BSnet(32, 1, istao=True),
    'depthnet': NormDepth(iskt=False),
    'refine_net': BSnet(64, 3, refine=True),
}

def norm_depth(depth):
    maxv = torch.max(depth)
    minv = torch.min(depth)
    n_depth = (depth-minv)/(maxv-minv)
    return n_depth

def testnature(net):
    net.eval()
    atonet.eval()
    refinenet.eval()

    torch.cuda.empty_cache()

    imgsavepath = './test_result/urhi/'
    naturepath = './test_data/urhi/'
    names = os.listdir(naturepath)


    for hname in names:

        filename = naturepath + hname

        img = pi.open(filename).convert('RGB')
        sizei = img.size

        h, w = (sizei[0] // 4) * 4, (sizei[1] // 4) * 4

        img = FF.crop(img,0,0,w,h)

        input = tfs.ToTensor()(img)
        input = torch.unsqueeze(input, dim=0)
        ohaze = input
        ohaze = ohaze.to(opt.device)
        tranp = net(ohaze, ohaze)
        atop = atonet(ohaze, ohaze)
        tranp = torch.threshold(tranp, 0.01, 0.01)
        dehazemap_0 = (ohaze - atop * (1 - tranp)) / tranp
        dehazemap = refinenet(dehazemap_0, ohaze)
        dehazemap = torch.clamp(dehazemap,0,1)
        vutils.save_image(dehazemap.cpu(), imgsavepath + hname)


def test_have_gt(datatype):
    net.eval()
    atonet.eval()
    refinenet.eval()

    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    hazypath = ''
    gtpath = ''
    if datatype == '4KID':
        hazypath = './test_data/4kID/hazy/'
        gtpath = './test_data/4kID/gt/'
    if datatype == 'SOT-IN':
        hazypath = './test_data/SOTIN/hazy/'
        gtpath = './test_data/SOTIN/gt/'


    hazylist = os.listdir(hazypath)
    for hname in hazylist:
        hazefullname = hazypath+hname
        gtfullname = gtpath+hname

        img = pi.open(hazefullname).convert('RGB')
        gt = pi.open(gtfullname).convert('RGB')

        input = tfs.ToTensor()(img)
        input = torch.unsqueeze(input, dim=0)
        ohaze = input
        ohaze = ohaze.to(opt.device)

        gt = tfs.ToTensor()(gt)
        gt = torch.unsqueeze(gt, dim=0)
        gt = gt
        gt = gt.to(opt.device)

        import time
        starttime = time.time()
        tranp = net(ohaze, ohaze)
        atop = atonet(ohaze, ohaze)
        tranp = torch.threshold(tranp, 0.01, 0.01)
        dehazemap_0 = (ohaze - atop * (1 - tranp)) / tranp
        dehazemap = refinenet(dehazemap_0, ohaze)
        endtime = time.time()
        ssim1 = ssim(dehazemap, gt).item()
        psnr1 = psnr(dehazemap, gt)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        print("运行时间:%.3f秒" % (endtime - starttime))
        scorestr = '==' + str(np.mean(ssims)) + '=====' + str(np.mean(psnrs))
        print(scorestr)
        vutils.save_image(dehazemap.cpu(),
                          './h4k/wre/'+hname)

if __name__ == "__main__":


    net = models_['trannet']
    net = net.to(opt.device)
    atonet = models_['atonet']
    atonet = atonet.to(opt.device)
    refinenet = models_['refine_net']
    depthnet = models_['depthnet']
    depthnet = depthnet.to(opt.device)
    refinenet = refinenet.to(opt.device)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(num_params / 1e6)

    if opt.device=='cuda':
        net=torch.nn.DataParallel(net)
        atonet = torch.nn.DataParallel(atonet)
        refinenet = torch.nn.DataParallel(refinenet)

    usestage1 = True
    stage1_pth_path = './pth_path/stage1/'
    stage2_pth_path = './pth_path/stage2/'
    if usestage1:
        pth_path = stage1_pth_path
    else:
        pth_path = stage2_pth_path

    net.load_state_dict(torch.load(pth_path+'Tnet.pth'))
    atonet.load_state_dict(torch.load(pth_path + 'Anet.pth'))
    refinenet.load_state_dict(torch.load(pth_path + 'Rnet.pth'))

    with torch.no_grad():
        testnature(net)





