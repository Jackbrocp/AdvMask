import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import get_training_dataloader, get_test_dataloader,get_cifar10_dataloader
import yaml
from warmup_scheduler import GradualWarmupScheduler

from Network import *
from adv_mask import *
from adv_mask import method_cifar100 as mv100
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
seed = 1000
setup_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-log_interval',type=int,default=50,help='log training status')
parser.add_argument('--sample_ratio',type=float,default=1.0,help="number of generated samples")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument('--conf', default='', type=str,  help=' yaml file')
args = parser.parse_args()
with open(args.conf) as f:
    cfg = yaml.safe_load(f)
print(cfg)
if cfg['writer']==True:
    writer = SummaryWriter('./Log/log_file')
best_acc = 0
best_epoch = 0
momentum = args.momentum
dataset = cfg['dataset']
net = get_model(cfg['model']['type'], cfg['dataset'])
net = torch.nn.DataParallel(net,device_ids=[0,1]).cuda()
if cfg['optimizer']['type'] == 'sgd':
    optimizer = optim.SGD(
        net.parameters(),
        lr=cfg['lr'],
        momentum=momentum,
        weight_decay=cfg['optimizer']['decay'],
        nesterov=cfg['optimizer']['nesterov']
    )
lr_schduler_type = cfg['lr_schedule']['type']
if lr_schduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epoch'], eta_min=0.)
elif lr_schduler_type == 'step':
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr_schedule']['milestones'],gamma=cfg['lr_schedule']['gamma'])
if cfg['lr_schedule']['warmup']!='' and  cfg['lr_schedule']['warmup']['epoch'] > 0:
    scheduler =  GradualWarmupScheduler(
        optimizer,
        multiplier = cfg['lr_schedule']['warmup']['multiplier'],
        total_epoch = cfg['lr_schedule']['warmup']['epoch'],
        after_scheduler = scheduler
    )
epoches = cfg['epoch']
batch = cfg['batch']
sample_ratio = args.sample_ratio
loss = nn.CrossEntropyLoss()
if dataset == 'cifar100':
    trainingloader = get_training_dataloader(
        num_workers=2,
        batch_size=batch ,
        shuffle=True,
        mask=cfg['mask']
    )
    testloader = get_test_dataloader(
        num_workers=2,
        batch_size=batch,
        shuffle=False
    )
elif dataset == 'cifar10':
    trainingloader, testloader = get_cifar10_dataloader(
        num_workers=2,
        batch_size=batch,
        shuffle=True,
        mask=cfg['mask']
    )
    
transform_after = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
])
def load_pointlist():
    print('=======================Load Point List==================')
    total_num=50000 
    point_list = []
    for i in tqdm(range(total_num)):
        if dataset == 'cifar100':
            p=np.load( './Attack_Mask/Attack_Mask_CIFAR100/point_list/{}.npy'.format(i))
        elif dataset == 'cifar10':
            p=np.load( './Attack_Mask/Attack_Mask_CIFAR10/point_list/{}.npy'.format(i))
        point_list.append(p)
    return point_list
if cfg['mask']== 'advmask':
    pointlist = load_pointlist()
def train(epoch):
    training_loss=0.0
    total = len(trainingloader.dataset)
    net.train()
    correct=0
    mask_num_list=[]
    global sample_ratio, epoches
    print('Sample ratio: {}'.format(sample_ratio))
    sample_ratio -= 0.8/epoches
    sample = 0
    if cfg['mask'] == 'advmask':
        if dataset == 'cifar100':
            stretagy = mv100.PointChoose()
        elif dataset == 'cifar10':
            stretagy = mv10.PointChoose()
    for i, (images, labels, filename) in enumerate(trainingloader):
        labels = labels.cuda()
        images = images.cuda()
        if  cfg['mask']== 'advmask':
            for j in range(len(images)):
                if np.random.rand()>sample_ratio:
                    sample+=1
                    mask = stretagy.createMask(mask_num_list,pointlist[int(filename[j][:-4])])
                    mask = torch.from_numpy(mask).cuda()
                    images[j] = images[j]*mask
        images = transform_after(images)
        optimizer.zero_grad()
        outputs = net(images)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()
        training_loss+=l.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        if (i+1)% args.log_interval==0:
            loss_mean = training_loss/(i+1)
            trained_total  = (i+1)*len(labels)
            acc = 100. * correct/trained_total
            progress = 100. * trained_total/total
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}  Acc: {:.6f} '.format(epoch,
                trained_total, total, progress, loss_mean, acc ))
    if cfg['writer'] == True:
        writer.add_scalar('trainng_loss',training_loss/len(trainingloader),epoch)
        writer.add_scalar('trainging_acc', 100. * correct/50000,epoch)
    print('Sample Numble: {}'.format(sample))
@torch.no_grad()
def eval_training(epoch=0 ):
    global best_acc
    global best_epoch
    net.eval()
    correct = 0.0
    total = 0.
    for (images, labels) in testloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        pred = torch.max(outputs.data,1)[1]
        total += labels.size(0)
        correct += (pred==labels).sum().item()
    accuracy = correct / total
    print('EPOCH:{}, ======================ACC:{}===================='.format(epoch, accuracy))
    if accuracy >= best_acc:
        best_acc = accuracy
        best_epoch = epoch
        state = {
            'net': net.state_dict(),
            'acc': accuracy,
            'epoch': epoch,
        }
        torch.save(state, '/data/yangsuorong/AdvMask_Model/{}_advmask.pth'.format(dataset))
    print('BEST EPOCH:{},BEST ACC:{}%'.format(best_epoch,best_acc*100.))
    if cfg['writer'] == True:
        writer.add_scalar('test_acc',accuracy,epoch)
 

if __name__ == '__main__':
    for epoch in tqdm(range(1, epoches + 1)):
        train(epoch)
        eval_training(epoch)
        scheduler.step()
        print('EPOCH{}:================Learning Rate:{}================='.format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    print('BEST ACC={}:{}\n'.format(best_epoch,best_acc))
        
