import sys
sys.path.append("..")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


import torch,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from network import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from stage2_dataload import *
print('log_dir :',log_dir)
print('model_name:',model_name)

models_={
    'tnet':BSnet(32,1,istran=True),
    'atonet':BSnet(32,1,istao=True),
    'depthnet':NormDepth(iskt=False),
    'refine_net':BSnet(64,3,refine=True,ismixup=False),
}

loaders_={
    'its_train':ITS_train_loader,
    'its_test':ITS_test_loader,
}


def norm_depth(depth):
    maxv = torch.max(depth)
    minv = torch.min(depth)
    n_depth = (depth-minv)/(maxv-minv)
    return n_depth

def tran2depth(tranp):
    # tranp = F.sigmoid(tranp)
    t1 = -torch.log(tranp)
    tdepth = norm_depth(t1)
    return tdepth

start_time=time.time()
T=opt.steps
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def get_dark_channel(I, w):

    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])

    return -dc


def train(net,loader_train,loader_test,optim,criterion):
    losses=[]
    start_step=0
    crloss = ContrastLoss()


    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp=torch.load(opt.model_dir)
        losses=ckp['losses']
        net.load_state_dict(ckp['model'])
        atonet.load_state_dict(ckp['atonet'])
        refinenet.load_state_dict(ckp['rnet'])
        start_step=ckp['step']
        max_ssim=ckp['max_ssim']
        max_psnr=ckp['max_psnr']
        print(f'start_step:{start_step} start training ---')
        print('maxpsnr:', max_psnr, 'max-ssim:', max_ssim)
    else :
        print('train from scratch *** ')

    for step in range(start_step+1,opt.steps+1):
        net.train()
        atonet.train()
        refinenet.train()
        depthnet.eval()

        lr=opt.lr
        if not opt.no_lr_sche:
            lr=lr_schedule_cosdecay(step,T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        unlabehaze,unlabel_gt,realc = next(iter(loader_train))
        unlabehaze = unlabehaze.to(opt.device)
        unlabel_gt = unlabel_gt.to(opt.device)
        realc = realc.to(opt.device)

        dehazemap = realc


        fake_tran_list = []
        depth_re = norm_depth(depthnet(dehazemap))
        betav = [0.5, 1, 1, 2, 2, 3, 3, 4]
        random.shuffle(betav)
        for k in range(depth_re.shape[0]):
            beta = random.uniform(betav[k], betav[k] + 1)
            depth_res = depth_re[k:k + 1, :, :, :]
            fake_trans = torch.exp(-beta * depth_res)
            fake_tran_list.append(fake_trans)

        fake_tran = torch.cat(fake_tran_list, dim=0)
        atolabel = torch.zeros_like(fake_tran)
        avalues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        random.shuffle(avalues)
        for i in range(atolabel.shape[0]):
            a_map = random.uniform(avalues[i], avalues[i] + 0.2)
            atolabel[i:i + 1, :, :, :] += a_map

        fake_haze = dehazemap * fake_tran + atolabel * (1 - fake_tran)
        atolabel = atolabel.to(opt.device)
        fake_haze = fake_haze.to(opt.device)
        fake_tran = fake_tran.to(opt.device)


        # Assembly data
        haze_total = torch.cat([fake_haze,unlabehaze],dim=0)
        clear_total = torch.cat([realc,unlabel_gt],dim=0)

        tranp = net(haze_total, haze_total)
        atop = atonet(haze_total, haze_total)
        tranpp = torch.threshold(tranp, 0.01, 0.01)
        dehazere = (haze_total - atop * (1 - tranpp)) / tranpp
        refinedehaze = refinenet(dehazere, haze_total)


        dc = get_dark_channel(haze_total, w=15)
        dc_shaped = dc.repeat(1, 3, 1, 1)
        onemask = torch.ones_like(dc_shaped)
        zeromask = torch.zeros_like(dc_shaped)
        masknosky = torch.where(dc_shaped>0.75,zeromask,onemask)
        masknosky = masknosky.to(opt.device)

        masksky = torch.where(dc_shaped>0.75,onemask,zeromask)
        masksky = masksky.to(opt.device)

        nosky1 = refinedehaze*masknosky
        nosky2 = clear_total*masknosky
        nosky3 = haze_total*masknosky


        mycrloss = crloss(nosky1, nosky2, nosky3)
        recloss = criterion[0](nosky1,nosky2)

        sky1 = haze_total*masksky
        sky2 = refinedehaze*masksky

        lwf_loss_sky = criterion[0](sky1,sky2)

        loss = mycrloss+lwf_loss_sky
        loss.backward()

        optim.step()
        optim.zero_grad()


        losses.append(loss.item())
        print(f'\rtrain loss : {lwf_loss_sky.item():.5f}|{mycrloss.item():.5f}|{recloss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

        if step % opt.eval_step ==0 :
            with torch.no_grad():
                pthpath = './save_stage2_pth_re/'+str(step)+'/'+'pth_save/'
                if not os.path.exists(pthpath):
                    os.makedirs(pthpath)
                torch.save(net.state_dict(), '%s/meta_dehaze_tnet_%d.pth' % (pthpath, step))
                torch.save(atonet.state_dict(), '%s/meta_dehaze_Anet_%d.pth' % (pthpath, step))
                torch.save(refinenet.state_dict(), '%s/meta_dehaze_rnet_%d.pth' % (pthpath, step))


import itertools
if __name__ == "__main__":
    loader_train=loaders_[opt.trainset]
    loader_test=loaders_[opt.testset]


    net=models_['tnet']
    net=net.to(opt.device)
    atonet = models_['atonet']
    atonet = atonet.to(opt.device)
    depthnet = models_['depthnet']
    depthnet = depthnet.to(opt.device)
    refinenet = models_['refine_net']
    refinenet = refinenet.to(opt.device)

    pth_path = './stage1_wights/'
    net.load_state_dict(torch.load(pth_path+'meta_dehaze_tnet_14000.pth'))
    atonet.load_state_dict(torch.load(pth_path + 'meta_dehaze_Anet_14000.pth'))
    refinenet.load_state_dict(torch.load(pth_path + 'meta_dehaze_rnet_14000.pth'))

    if opt.device=='cuda':
        net=torch.nn.DataParallel(net)
        atonet = torch.nn.DataParallel(atonet)
        refinenet = torch.nn.DataParallel(refinenet)
        cudnn.benchmark=True

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))

    optimizer = optim.Adam(itertools.chain(net.parameters(),atonet.parameters(),refinenet.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()
    train(net,loader_train,loader_test,optimizer,criterion)


