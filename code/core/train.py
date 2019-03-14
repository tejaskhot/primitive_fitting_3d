import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from model import *
from datasets import *
from tensorboardX import SummaryWriter
from hungarian_loss import *
import time
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--server', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--model_name', type=str, default='norot_rnn1',  help='model name')

parser.add_argument('--num_data', type=int, default=None, help='number of input data instances')
parser.add_argument('--num_points', type=int, default=4096, help='number of points')
parser.add_argument('--num_primitives', type=int, default=3, help='number of cuboids')
parser.add_argument('--num_params', type=int, default=6, help='number of parameters to predict per primitive')
parser.add_argument('--num_channels', type=int, default=3, help='Number of input channels')
parser.add_argument('--num_faces', type=int, default=6, help='number of faces per primitive')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')

parser.add_argument('--lr', type=float, default=1e-3,help='learning rate')
parser.add_argument('--alpha', type=float, default=0.1,help='alpha for ce loss weight')
parser.add_argument('--grad_norm_limit', type=float, default=1,help='grad norm clipping limit')

parser.add_argument('--save_freq', type=int, default=50, help='save model to file after these many epochs')
parser.add_argument('--print_freq', type=int, default=10, help='print intermediate losses after these many epochs')

parser.add_argument('--data_path', type=str, default = '/scratch/tkhot/lod/pcd_ply/',  help='data path')
parser.add_argument('--params_path', type=str, default = '/scratch/tkhot/lod/params.npy',  help='params path')
parser.add_argument('--outf', type=str, default='',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--logs', type=str, default = '',  help='logs path')
parser.add_argument('--imdir', type=str, default = '',  help='logs path')


opt = parser.parse_args()
green = lambda x:"\033[92m" + x + "\033[00m"
print(opt)

if not opt.outf: opt.outf = '../outputs/saved_models'
if not opt.title: opt.title = opt.model_name
if not opt.logs: opt.logs = '../outputs/'
if not opt.imdir: opt.imdir = '../plots/{}'.format(opt.model_name)
if opt.num_data:
    if opt.num_data < opt.batch_size: opt.batch_size=opt.num_data

writer = SummaryWriter(opt.logs+opt.model_name)

# opt.manualSeed = random.randint(1, 10000) # fix seed
opt.manualSeed = 1
print('Random Seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# DECAY_STEP = 800000.
DECAY_STEP = 15000
DECAY_RATE = 0.1

dataset_train = BuildingDataloader(data_path=opt.data_path, params_path=opt.params_path, dtype='train', num_data=opt.num_data, num_params=opt.num_params, num_channels=opt.num_channels, num_primitives=opt.num_primitives, batch_size=opt.batch_size)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers), pin_memory=True)
dataset_test = BuildingDataloader(data_path=opt.data_path, params_path=opt.params_path, dtype='test', num_data=opt.num_data, num_params=opt.num_params, num_channels=opt.num_channels, num_primitives=opt.num_primitives, batch_size=opt.batch_size)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers), pin_memory=True)

print('len(dataset_train) : ', len(dataset_train))
print('len(dataset_test) : ', len(dataset_test))
try:
    os.makedirs(opt.outf)
except OSError:
    pass

model = RLNet(num_points=opt.num_points, out_size=opt.num_params, num_primitives=opt.num_primitives)
model.cuda()

if opt.model != '':
    model.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(model.parameters(), lr=opt.lr, amsgrad=True)

criterion = GeometricLoss(batch_size=1, samples_per_face=50, num_faces=opt.num_faces,  num_params=opt.num_params)
criterion.cuda()

ce_crit = nn.CrossEntropyLoss()
hungarian_loss = HungarianLoss()
hungarian_loss.cuda()

def adjust_lr(optim):
    for g in optim.param_groups:
        g['lr'] = 0.75*g['lr']
        print('lr changed to {}', g['lr'])

print('-'*50)
alpha = 0.5
for epoch in range(opt.nepoch):

    if epoch>0:
        print(green('[{}] Loss : {}'.format(epoch+1, np.mean(total_losses))))
        if ((epoch+1) % 75) == 0:
            adjust_lr(optimizer)
    total_losses = []
    for i, data in enumerate(dataloader_train,0):
        points, gt_params, gt_probs, gt_counts = data
        points, gt_params, gt_probs, gt_counts = Variable(points), Variable(gt_params), Variable(gt_probs.type(torch.LongTensor)), Variable(gt_counts.type(torch.LongTensor))
        points = points.transpose(2,1)
        points, gt_params, gt_probs, gt_counts = points.cuda(), gt_params.cuda(), gt_probs.cuda(), gt_counts.cuda()

        del data

        optimizer.zero_grad()
        pred, probs = model(points)
        out = torch.reshape(pred, (opt.batch_size, opt.num_primitives, opt.num_params))
        mse_loss = hungarian_loss(out, gt_params, gt_counts)
        ce_loss = ce_crit(probs, gt_probs)
        total_loss = mse_loss + opt.alpha*ce_loss
        total_loss.backward()
        if opt.grad_norm_limit:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm_limit)
        optimizer.step()
        total_losses.append(total_loss.data)

        writer.add_scalar('mse_loss', mse_loss, epoch*len(dataloader_train)+i)
        writer.add_scalar('ce_loss', ce_loss, epoch*len(dataloader_train)+i)
        writer.add_scalar('total_loss', total_loss, epoch*len(dataloader_train)+i)
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         writer.add_scalar(name+"_grad", torch.norm(param.grad), epoch*len(dataloader_train)+i)

        # free up some space -- dataloader first allocates memory and then assigns; so deleting ensures we don't need double the memory
        del points, gt_params, gt_probs, pred, probs

    if (epoch+1)%opt.save_freq==0:
        torch.save(model.state_dict(), '{}/{}_{}.pth'.format(opt.outf, opt.model_name, epoch+1))

if opt.nepoch>0:
    torch.save(model.state_dict(), '{}/{}_{}.pth'.format(opt.outf, opt.model_name, epoch+1))
