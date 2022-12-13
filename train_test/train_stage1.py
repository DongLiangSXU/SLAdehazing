import sys
sys.path.append("..")
import os

# if you have more than one GPU, you can set it to '0,1,2,....'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
from stage1_dataload import *
print('log_dir :',log_dir)
print('model_name:',model_name)

models_={
    'tnet':BSnet(32,1,istran=True),
    'atonet':BSnet(32,1,istao=True),
    '''
    NormDepth code from https://github.com/ialhashim/DenseDepth
    But, We made a pytorch version
    you can get the pytorch version weights from links below:
    https://drive.google.com/drive/folders/1xq1tg7wvNJeZTw8w4RnqEsJzQRcmHLCI?usp=sharing
    When training, we only use indoor version weights. Because There is overfitting of the parameters trained on Kitti.
    '''
    'depthnet':NormDepth(iskt=False),
    'refine_net':BSnet(64,3,refine=True,ismixup=True),
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


def train(net,loader_train,loader_test,optim,criterion):
    losses=[]
    start_step=0
    max_ssim=0
    max_psnr=0
    ssims=[]
    psnrs=[]
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
        psnrs=ckp['psnrs']
        ssims=ckp['ssims']
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

        clear = next(iter(loader_train))

        clear=clear.to(opt.device)
        dehazemap = clear

        # depthnet make-haze

        fake_tran_list = []
        depth_re = norm_depth(depthnet(dehazemap))
        betav = [0.5,1,1,2,2,3,3,4]
        random.shuffle(betav)
        for k in range(depth_re.shape[0]):
            beta = random.uniform(betav[k], betav[k]+1)
            depth_res = depth_re[k:k+1,:,:,:]
            fake_trans = torch.exp(-beta * depth_res)
            fake_tran_list.append(fake_trans)

        fake_tran = torch.cat(fake_tran_list,dim=0)
        atolabel = torch.zeros_like(fake_tran)
        avalues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        random.shuffle(avalues)
        for i in range(atolabel.shape[0]):
            a_map = random.uniform(avalues[i],avalues[i]+0.2)
            atolabel[i:i+1,:,:,:] += a_map

        fake_haze = dehazemap*fake_tran+atolabel*(1-fake_tran)
        atolabel = atolabel.to(opt.device)
        fake_haze = fake_haze.to(opt.device)
        fake_tran = fake_tran.to(opt.device)

        # train-net

        tranp = net(fake_haze,fake_haze)
        atop = atonet(fake_haze,fake_haze)
        tranpp = torch.threshold(tranp,0.01,0.01)
        dehazere = (fake_haze - atop * (1 - tranpp)) / tranpp
        refinedehaze = refinenet(dehazere,fake_haze)

        loss= criterion[0](tranp, fake_tran)
        lossato = criterion[0](atop,atolabel)

        mycrloss = crloss(refinedehaze,clear,fake_haze)


        if step<500:
            lossa = loss + lossato
            lossa.backward()
        else:
            lossa = mycrloss+criterion[0](dehazere, clear)+criterion[0](refinedehaze, clear)
            lossa.backward()

        optim.step()
        optim.zero_grad()


        losses.append(loss.item())
        print(f'\rtrain loss : {lossa.item():.5f}|{loss.item():.5f}|{lossato.item():.5f}|{mycrloss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

        if step % opt.eval_step ==0 :
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)
                pthpath = './save_weights/'+str(step)+'/'+'pth_save/'
                if not os.path.exists(pthpath):
                    os.makedirs(pthpath)

                torch.save(net.state_dict(), '%s/meta_dehaze_tnet_%d.pth' % (pthpath, step))
                torch.save(atonet.state_dict(), '%s/meta_dehaze_Anet_%d.pth' % (pthpath, step))
                torch.save(refinenet.state_dict(), '%s/meta_dehaze_rnet_%d.pth' % (pthpath, step))

                print(
                f'\nstep :{0} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}|max_ssim:{max_ssim:.4f}| max_psnr:{max_psnr:.4f}')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr :
                max_ssim=max(max_ssim,ssim_eval)
                max_psnr=max(max_psnr,psnr_eval)
                torch.save({
                    'step':step,
                    'max_psnr':max_psnr,
                    'max_ssim':max_ssim,
                    'ssims':ssims,
                    'psnrs':psnrs,
                    'losses':losses,
                    'atonet':atonet.state_dict(),
                    'rnet':refinenet.state_dict(),
                    'model':net.state_dict(),
                },opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

#
def test(net,loader_test):
    net.eval()
    atonet.eval()
    refinenet.eval()
    torch.cuda.empty_cache()
    ssims=[]
    psnrs=[]
    for i ,(haze, clear,ohaze) in enumerate(loader_test):
        ohaze = ohaze.to(opt.device)
        tranp = net(ohaze,ohaze)
        atop = atonet(ohaze,ohaze)
        tranp = torch.threshold(tranp, 0.01, 0.01)
        dehazemap_0 = (ohaze - atop * (1 - tranp)) / tranp
        dehazemap = refinenet(dehazemap_0,ohaze)
        y = clear.to(opt.device)
        ssim1=ssim(dehazemap,y).item()
        psnr1=psnr(dehazemap,y)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    return np.mean(ssims) ,np.mean(psnrs)


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


