import importlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import time
from datetime import datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch import nn as nn
from torch.autograd import grad
from torch.nn import functional as F


def get_free_gpu(num: int = None, usage_threshold=1.0, verbose=True, return_str=True):
    if (num is not None and num <= 0) or usage_threshold > 1.0 or usage_threshold < 0.0:
        raise ValueError

    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.free,memory.total"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode('utf8')),
                         names=['memory.free', 'memory.total'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    gpu_df['memory.total'] = gpu_df['memory.total'].map(lambda x: int(x.rstrip(' [MiB]')))
    gpu_df = gpu_df.sort_values(by='memory.free', ascending=False)
    if verbose:
        print('GPU usage:\n{}'.format(gpu_df))

    gpu_usages = 1 - gpu_df['memory.free'] / gpu_df['memory.total']

    free_gpus = []
    for i in range(len(gpu_usages)):
        if gpu_usages.iloc[i] < usage_threshold:
            free_gpus.append(gpu_usages.index[i])

    if num is not None:
        if len(gpu_df) < num:
            raise RuntimeError('No enough GPU')
        free_gpus = free_gpus[:num]
    if verbose:
        for gpu in free_gpus:
            print('Returning GPU{}'.format(gpu))

    if return_str:
        return ','.join(str(x) for x in free_gpus)
    return free_gpus


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if output.dim() == 1:  # binary acc
            correct = torch.eq(torch.round(output).type(target.type()), target).view(-1)
            return [correct.sum() / correct.shape[0]]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, filename='checkpoint.pth', logdir=None):
    path = os.path.join(logdir if logdir is not None else os.getcwd(), filename)
    torch.save(state, path)


def save_result_dict(result_dict, logdir, filename='results.json'):
    with open(os.path.join(logdir, filename), 'w') as fp:
        json.dump(result_dict, fp, indent=2)


def get_network(arch, *args, **kwargs):
    class_name = None
    if '.' in arch:
        arch, class_name = arch.strip().split('.')
    module = importlib.import_module('models.' + arch)

    if class_name is not None:
        model = getattr(module, class_name)
    else:
        raise NotImplementedError

    return model(*args, **kwargs)


def bregman_divergence(r_true, r_pred, type='bkl'):
    if type not in ('bkl', 'sq', 'ukl'):
        raise ValueError("Invalid Bregman type")
    if type =='bkl':
        return (1 + r_true) * ((1 + r_pred) / (1 + r_true)).log() + r_true * (r_true / r_pred).log()
    elif type == 'sq':
        return (r_pred - r_true).square() * 0.5
    else:
        return r_true * (r_true / r_pred).log() - r_true + r_pred

def kl_normal_standardnormal(mu, std):
    return 0.5 * (mu.pow(2) + std.pow(2) - 2 * std.log() - 1).sum(1)


def nonlinear_ib_term(mean_t, std):
    def compute_distances(x):
        x_norm = (x**2).sum(1).view(-1,1)
        x_t = torch.transpose(x,0,1)
        x_t_norm = x_norm.view(1,-1)
        dist = x_norm + x_t_norm - 2.0*torch.mm(x,x_t)
        dist = torch.clamp(dist,0,np.inf)

        return dist

    def KDE_IXT_estimation(var, mean_t):
        n_batch, d = mean_t.shape

        # calculation of the constant
        normalization_constant = math.log(n_batch)

        # calculation of the elements contribution
        dist = compute_distances(mean_t)
        distance_contribution = - torch.mean(torch.logsumexp(input=- 0.5 * dist / var,dim=1))

        # mutual information calculation (natts)
        I_XT = normalization_constant + distance_contribution 

        return I_XT

    IXT = KDE_IXT_estimation(std, mean_t) # in natts
    return IXT / np.log(2) 

def save_current_code(args, filepath=__file__):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    filepath = Path(filepath)
    shutil.copyfile(filepath.absolute(), os.path.join(args.logdir, filepath.name))
    model_filename = args.arch.strip().split('.')[0] + '.py'
    shutil.copyfile(filepath.parent.joinpath('models', model_filename), os.path.join(args.logdir, model_filename))

def pairwise_distances(x):
    #x should be two dimensional
    x = x.flatten(1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)

def reyi_entropy(x,sigma):
    alpha = 1.01
    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy


def deterministic_ib_term(x,y,s_x,s_y):
    
    Hx = reyi_entropy(x,sigma=s_x)
    Hy = reyi_entropy(y,sigma=s_y)
    Hxy = joint_entropy(x,y,s_x,s_y)
    Ixy = Hx+Hy-Hxy
    #normlize = Ixy/(torch.max(Hx,Hy))
    
    return Ixy

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def validate(loader, model, args, epoch=None, writer=None, tag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1],
        prefix=f'{tag.capitalize()}: ')
    ce_loss_func = nn.CrossEntropyLoss()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            logits = model(images)
            loss = ce_loss_func(logits, target)
            # measure accuracy and record loss
            acc1, = accuracy(logits, target, topk=(1,))

            bs = images.size(0)
            losses.update(loss.item(), bs)
            top1.update(acc1[0].item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display_summary()

    if writer is not None:
        writer.add_scalar(f'Time/{tag}', batch_time.avg, epoch)
        writer.add_scalar(f'Losses/{tag}', losses.avg, epoch)
        writer.add_scalar(f'Accuracy1/{tag}', top1.avg, epoch)
    return top1.avg


def train_rib(train_loader, ghost_loader, model, model_cri, optimizer, optimizer_cri, scheduler, scheduler_cri,
              args,
              epoch, writer=None, bregman_type='bkl'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    bregman_losses = AverageMeter('BregmanLoss', ':.4e')
    critic_losses = AverageMeter('CriticLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    critic_top1 = AverageMeter('CriticAcc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, ce_losses, bregman_losses, critic_losses, top1, critic_top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    model_cri.train()

    end = time.time()
    r_true = torch.ones(1).cuda()
    for i, ((images, target), (ghost_images, _)) in enumerate(zip(train_loader, ghost_loader)):
        bs = images.size(0)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ghost_images = ghost_images.cuda(non_blocking=True)

        # compute output
        logits, feat = model(images, with_feat=True)
        logits_ghost, feat_ghost = model(ghost_images, with_feat=True)
        feat_all = torch.stack([feat, feat_ghost], dim=1)
        mask_target = torch.randint(2, size=(bs,), device=feat_all.device)
        if not torch.any(mask_target):
            continue
        feat_a = feat_all[torch.arange(bs), mask_target]
        feat_b = feat_all[torch.arange(bs), 1 - mask_target]
        feat_all = torch.cat([feat_b, feat_a], dim=1)
        log_r_pred = model_cri(feat_all)

        cls_loss = F.cross_entropy(logits, target)
        bregman_loss = bregman_divergence(r_true, log_r_pred.exp(), type=bregman_type).mean() * args.beta
        loss = cls_loss + bregman_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        log_r_pred = model_cri(feat_all.detach())
        critic_loss = (F.softplus(-log_r_pred[mask_target == 1])).mean() + F.softplus(log_r_pred).mean()
        critic_loss = critic_loss * args.lam
        # compute gradient and do SGD step
        optimizer_cri.zero_grad()
        critic_loss.backward()
        optimizer_cri.step()
        scheduler_cri.step()

        # measure accuracy and record loss
        acc1, = accuracy(logits, target, topk=(1,))
        critic_acc1, = accuracy(torch.sigmoid(log_r_pred), mask_target, topk=(1,))
        losses.update(loss.item(), bs)
        ce_losses.update(cls_loss.item(), bs)
        bregman_losses.update(bregman_loss.item(), bs)
        critic_losses.update(critic_loss.item(), bs)
        top1.update(acc1[0].item(), bs)
        critic_top1.update(critic_acc1.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")
    progress.display_summary()

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalar('Losses/train', losses.avg, epoch)
        writer.add_scalar('CELosses/train', ce_losses.avg, epoch)
        writer.add_scalar('BregmanLosses/train', bregman_losses.avg, epoch)
        writer.add_scalar('CriticLosses/train', critic_losses.avg, epoch)
        writer.add_scalar('Accuracy1/train', top1.avg, epoch)
        writer.add_scalar('CriticAccuracy1/train', critic_top1.avg, epoch)
    return top1.avg


def train_rib_minimax(train_loader, ghost_loader, model, model_cri, optimizer, optimizer_cri, scheduler, scheduler_cri,
                      args, epoch, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    enc_losses = AverageMeter('EncLoss', ':.4e')
    critic_losses = AverageMeter('CriticLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    critic_top1 = AverageMeter('CriticAcc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, ce_losses, enc_losses, critic_losses, top1, critic_top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    model_cri.train()

    end = time.time()
    # r_true = torch.ones(1).cuda()
    for i, ((images, target), (ghost_images, _)) in enumerate(zip(train_loader, ghost_loader)):
        bs = images.size(0)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ghost_images = ghost_images.cuda(non_blocking=True)

        # compute output
        logits, feat = model(images, with_feat=True)
        logits_ghost, feat_ghost = model(ghost_images, with_feat=True)
        feat_all = torch.stack([feat, feat_ghost], dim=1)
        mask_target = torch.randint(2, size=(bs,), device=feat_all.device)
        if not torch.any(mask_target):
            continue
        feat_a = feat_all[torch.arange(bs), mask_target]
        feat_b = feat_all[torch.arange(bs), 1 - mask_target]
        feat_all = torch.cat([feat_b, feat_a], dim=1)
        log_r_pred = model_cri(feat_all)

        cls_loss = F.cross_entropy(logits, target)
        enc_loss = (F.softplus(-log_r_pred[mask_target == 1])).mean() + F.softplus(log_r_pred).mean()
        enc_loss = enc_loss * -args.beta
        loss = cls_loss + enc_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        log_r_pred = model_cri(feat_all.detach())
        critic_loss = (F.softplus(-log_r_pred[mask_target == 1])).mean() + F.softplus(log_r_pred).mean()
        critic_loss = critic_loss * args.lam
        # compute gradient and do SGD step
        optimizer_cri.zero_grad()
        critic_loss.backward()
        optimizer_cri.step()
        scheduler_cri.step()

        # measure accuracy and record loss
        acc1, = accuracy(logits, target, topk=(1,))
        critic_acc1, = accuracy(torch.sigmoid(log_r_pred), mask_target, topk=(1,))
        losses.update(loss.item(), bs)
        ce_losses.update(cls_loss.item(), bs)
        enc_losses.update(enc_loss.item(), bs)
        critic_losses.update(critic_loss.item(), bs)
        top1.update(acc1[0].item(), bs)
        critic_top1.update(critic_acc1.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")
    progress.display_summary()

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalar('Losses/train', losses.avg, epoch)
        writer.add_scalar('CELosses/train', ce_losses.avg, epoch)
        writer.add_scalar('EncLosses/train', enc_losses.avg, epoch)
        writer.add_scalar('CriticLosses/train', critic_losses.avg, epoch)
        writer.add_scalar('Accuracy1/train', top1.avg, epoch)
        writer.add_scalar('CriticAccuracy1/train', critic_top1.avg, epoch)
    return top1.avg


def train_critic(train_loader, ghost_loader, model, model_cri, optimizer_cri, scheduler, args, epoch, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    critic_losses = AverageMeter('CriticLoss', ':.4e')
    critic_top1 = AverageMeter('CriticAcc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, critic_losses, critic_top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.eval()
    model_cri.train()

    end = time.time()

    for i, ((train_images, _), (ghost_images, _)) in enumerate(zip(train_loader, ghost_loader)):
        bs = train_images.size(0)

        train_images = train_images.cuda(non_blocking=True)
        ghost_images = ghost_images.cuda(non_blocking=True)

        # compute output
        logits_train, feat_train = model(train_images, with_feat=True)
        logits_ghost, feat_ghost = model(ghost_images, with_feat=True)
        feat_all = torch.stack([feat_train, feat_ghost], dim=1)
        mask_target = torch.randint(2, size=(bs,), device=feat_all.device)
        if not torch.any(mask_target):
            continue
        feat_a = feat_all[torch.arange(bs), mask_target]
        feat_b = feat_all[torch.arange(bs), 1 - mask_target]
        feat_all = torch.cat([feat_b, feat_a], dim=1)
        log_r_pred = model_cri(feat_all.detach())
        critic_loss = (F.softplus(-log_r_pred[mask_target == 1])).mean() + F.softplus(log_r_pred).mean()

        optimizer_cri.zero_grad()
        critic_loss.backward()
        optimizer_cri.step()
        if scheduler is not None:
            scheduler.step()

        # measure accuracy and record loss
        mask_acc1, = accuracy(torch.sigmoid(log_r_pred), mask_target, topk=(1,))
        critic_losses.update(critic_loss.item(), bs)
        critic_top1.update(mask_acc1.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(critic_loss).any():
            raise RuntimeError("nan in loss!")
    progress.display_summary()

    if writer is not None:
        writer.add_scalar('CriticLosses/train', critic_losses.avg, epoch)
        writer.add_scalar('CriticAccuracy1/train', critic_top1.avg, epoch)
    return critic_top1.avg


def logmeanexp(x):
    size = x.size(0)
    logsumexp = torch.logsumexp(x, dim=0)
    return logsumexp - torch.log(torch.tensor(size, device=x.device))


def estimate_recog(sup_loader, mask, model, model_cri, args):
    # switch to train mode
    model.eval()
    model_cri.eval()
    lor_r_pred_list = []
    with torch.no_grad():
        for i, (images, _) in enumerate(sup_loader):
            bs = images.size(0) // 2

            if not images.is_cuda:
                images = images.cuda(non_blocking=True)

            # compute output
            logits, feat = model(images, with_feat=True)
            feat = feat.reshape(bs, -1)
            log_r_pred = model_cri(feat)
            lor_r_pred_list.append(log_r_pred.detach().cpu())
        log_r_pred = torch.cat(lor_r_pred_list, dim=0).squeeze()
        recog_est = torch.sigmoid(log_r_pred)
    return recog_est.numpy()


def estimate_recog_and_get_prediction(sup_loader, mask, model, model_cri, args):
    # switch to train mode
    model.eval()
    model_cri.eval()
    lor_r_pred_list = []
    final_logits = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(sup_loader):
            bs = images.size(0) // 2

            if not images.is_cuda:
                images = images.cuda(non_blocking=True)

            # compute output
            logits, feat = model(images, with_feat=True)
            final_logits.append(logits.cpu())
            feat = feat.reshape(bs, -1)
            log_r_pred = model_cri(feat)
            lor_r_pred_list.append(log_r_pred.detach().cpu())
        final_logits = torch.cat(final_logits, dim=0)
        log_r_pred = torch.cat(lor_r_pred_list, dim=0).squeeze()
        recog_est = torch.sigmoid(log_r_pred)
    return recog_est.numpy(), final_logits

def train_baseline(train_loader, model, optimizer, scheduler, args, epoch, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, ce_losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        bs = images.size(0)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        logits = model(images)

        cls_loss = F.cross_entropy(logits, target)
        loss = cls_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # measure accuracy and record loss
        acc1, = accuracy(logits, target, topk=(1,))
        losses.update(loss.item(), bs)
        ce_losses.update(cls_loss.item(), bs)
        top1.update(acc1[0].item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")
    progress.display_summary()

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalar('Losses/train', losses.avg, epoch)
        writer.add_scalar('CELosses/train', ce_losses.avg, epoch)
        writer.add_scalar('Accuracy1/train', top1.avg, epoch)
    return top1.avg


def train_pib(train_loader, model, energy_decay, optimizer, scheduler, args, epoch, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    iiw_losses = AverageMeter('IIWLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, ce_losses, iiw_losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        bs = images.size(0)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        logits = model(images)

        cls_loss = F.cross_entropy(logits, target)
        loss = cls_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        if epoch > 0:
            energy_decay.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # measure accuracy and record loss
        acc1, = accuracy(logits, target, topk=(1,))
        losses.update(loss.item(), bs)
        ce_losses.update(cls_loss.item(), bs)
        iiw_losses.update(energy_decay.item())
        top1.update(acc1[0].item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")
    progress.display_summary()

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalar('Losses/train', losses.avg, epoch)
        writer.add_scalar('CELosses/train', ce_losses.avg, epoch)
        writer.add_scalar('IIWLosses/train', iiw_losses.avg, epoch)
        writer.add_scalar('Accuracy1/train', top1.avg, epoch)
    return top1.avg


def train_vib(train_loader, model, optimizer, scheduler, args, epoch, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    kl_losses = AverageMeter('KLVFLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, ce_losses, kl_losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        bs = images.size(0)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        logits, (mu, std) = model(images, with_feat=True)
        cls_loss = F.cross_entropy(logits, target)
        kl_loss = kl_normal_standardnormal(mu, std).mean() * args.beta
        loss = cls_loss + kl_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # measure accuracy and record loss
        acc1, = accuracy(logits, target, topk=(1,))
        losses.update(loss.item(), bs)
        ce_losses.update(cls_loss.item(), bs)
        kl_losses.update(kl_loss.item(), bs)
        top1.update(acc1[0].item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")
    progress.display_summary()

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalar('Losses/train', losses.avg, epoch)
        writer.add_scalar('CELosses/train', ce_losses.avg, epoch)
        writer.add_scalar('KLLosses/train', kl_losses.avg, epoch)
        writer.add_scalar('Accuracy1/train', top1.avg, epoch)
    return top1.avg

def train_nib(train_loader, model, optimizer, scheduler, args, epoch, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    kl_losses = AverageMeter('KLVFLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, ce_losses, kl_losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        bs = images.size(0)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        logits, (mu, std) = model(images, with_feat=True)
        cls_loss = F.cross_entropy(logits, target)
        kl_loss = nonlinear_ib_term(mu, std) * args.beta
        loss = cls_loss + kl_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # measure accuracy and record loss
        acc1, = accuracy(logits, target, topk=(1,))
        losses.update(loss.item(), bs)
        ce_losses.update(cls_loss.item(), bs)
        kl_losses.update(kl_loss.item(), bs)
        top1.update(acc1[0].item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")
    progress.display_summary()

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalar('Losses/train', losses.avg, epoch)
        writer.add_scalar('CELosses/train', ce_losses.avg, epoch)
        writer.add_scalar('KLLosses/train', kl_losses.avg, epoch)
        writer.add_scalar('Accuracy1/train', top1.avg, epoch)
    return top1.avg

def train_dib(train_loader, model, optimizer, scheduler, args, epoch, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    kl_losses = AverageMeter('KLVFLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, ce_losses, kl_losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        bs = images.size(0)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        logits, feature = model(images, with_feat=True)
        cls_loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            k = torch.cdist(feature.unsqueeze(0), feature.unsqueeze(0)).squeeze()
            sigma_z = torch.mean(torch.sort(k[:, :10], 1)[0])
            
            # Z_numpy = feature.cpu().detach().numpy()
            # k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            # sigma_z = np.mean(np.mean(np.sort(k[:, :10], 1))) 
            # print(sigma_z_tr, sigma_z)
            # assert torch.allclose(sigma_z_tr, torch.tensor(sigma_z, dtype=sigma_z_tr.dtype).to(sigma_z_tr.device))

            k_input = torch.cdist(images.unsqueeze(0), images.unsqueeze(0)).squeeze()
            sigma_input = torch.mean(torch.sort(k_input[:, :10], 1)[0])
            # inputs_numpy = images.cpu().detach().numpy()
            # inputs_numpy = inputs_numpy.reshape(bs,-1)
            # k_input = squareform(pdist(inputs_numpy, 'euclidean'))
            # sigma_input = np.mean(np.mean(np.sort(k_input[:, :10], 1)))
            # assert torch.allclose(sigma_input_tr, torch.tensor(sigma_input, dtype=sigma_input_tr.dtype).to(device=sigma_input_tr.device))
            
        kl_loss = deterministic_ib_term(images, feature, s_x=sigma_input,s_y=sigma_z) * args.beta
        loss = cls_loss + kl_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # measure accuracy and record loss
        acc1, = accuracy(logits, target, topk=(1,))
        losses.update(loss.item(), bs)
        ce_losses.update(cls_loss.item(), bs)
        kl_losses.update(kl_loss.item(), bs)
        top1.update(acc1[0].item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")
    progress.display_summary()

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalar('Losses/train', losses.avg, epoch)
        writer.add_scalar('CELosses/train', ce_losses.avg, epoch)
        writer.add_scalar('KLLosses/train', kl_losses.avg, epoch)
        writer.add_scalar('Accuracy1/train', top1.avg, epoch)
    return top1.avg


def get_scheduler(args, optimizer, T_max=None):
    if args.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    elif args.lr_policy == 'milestones':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.lr_policy == 'cosine':
        assert T_max is not None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    else:
        raise ValueError('Unknown LR scheduler')
    return scheduler


class Task(object):

    def __init__(self, name, cmd, logdir='.'):
        self.name = name
        self.cmd = cmd
        self.proc = None
        self.add_date = datetime.now()
        self.finish_date = None
        self.duration = None
        self.gpu = None
        self.filename = os.path.join(logdir, name + '.txt')

    def assign_gpu(self, gpu):
        self.gpu = gpu

    def start(self, gpu=None):
        self.gpu = gpu if gpu is not None else [0]
        self.proc = self.run_command(self.cmd, self.gpu, self.filename)

    def wait(self):
        self.proc.wait()
        self.finish_date = datetime.now()
        self.duration = self.finish_date - self.add_date
        return self.duration

    def is_stop(self):
        return self.proc is None or self.proc.poll() is not None

    def __repr__(self):
        return '{}::GPU{}::{}'.format(self.name, self.gpu, ' '.join(self.cmd))

    @staticmethod
    def run_command(cmd, gpu, filename):
        pipe = open(filename, 'w')
        pipe.write(' '.join(cmd) + '\n')
        proc = subprocess.Popen(cmd,
                                stdout=pipe,
                                stderr=subprocess.STDOUT,
                                env={"CUDA_VISIBLE_DEVICES": ','.join(str(g) for g in gpu)},
                                shell=False)
        return proc


def wait_gpu(num, waitsecs, usage_threshold, reserve_gpus=None):
    if num <= 0:
        raise ValueError

    first_print = True
    while True:
        free_gpu_ids = get_free_gpu(usage_threshold=usage_threshold, verbose=False, return_str=False)
        if reserve_gpus is not None and len(reserve_gpus) > 0:
            free_gpu_ids = [i for i in free_gpu_ids if i not in reserve_gpus]
        if len(free_gpu_ids) < num:
            print("=> waiting GPU." if first_print else ".", end='')
            first_print = False
            sleep(waitsecs)
        else:
            print('=> found GPU {}'.format(free_gpu_ids))
            return free_gpu_ids[:num]


def discrete_mi_est(xs, ys, nx=2, ny=2):
    prob = np.zeros((nx, ny))
    for a, b in zip(xs, ys):
        prob[a, b] += 1.0 / len(xs)
    pa = np.sum(prob, axis=1)
    pb = np.sum(prob, axis=0)
    mi = 0
    for a in range(nx):
        for b in range(ny):
            if prob[a, b] < 1e-9:
                continue
            mi += prob[a, b] * np.log(prob[a, b] / (pa[a] * pb[b]))
    return max(0.0, mi)


def get_prediction(loader, model):
    model.eval()
    final_logits = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            images = data[0]
            if images.dim() == 5:
                images = images.view(images.shape[0] * 2, *images.shape[2:])
            if not images.is_cuda:
                images = images.cuda(non_blocking=True)

            logits = model(images)
            final_logits.append(logits.cpu())

    final_logits = torch.cat(final_logits, dim=0)
    return final_logits


def save_prediction(pred, filename, logdir):
    path = os.path.join(logdir if logdir is not None else os.getcwd(), filename)
    with open(path, "wb") as f:
        pickle.dump(pred, f)


def load_prediction(filename, logdir):
    path = os.path.join(logdir if logdir is not None else os.getcwd(), filename)
    with open(path, "rb") as f:
        pred = pickle.load(f)
    return pred


def estimate_fcmi_bound(preds, masks, num_classes, num_examples):
    bound = 0.0
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        ps = [p[2 * idx:2 * idx + 2] for p in preds]
        for i in range(len(ps)):
            ps[i] = torch.argmax(ps[i], dim=1)
            ps[i] = num_classes * ps[i][0] + ps[i][1]
            ps[i] = ps[i].item()
        cur_mi = discrete_mi_est(ms, ps, nx=2, ny=num_classes ** 2)
        bound += np.sqrt(2 * cur_mi)

    bound *= 1 / num_examples

    return bound


def compute_information_bp_fast(model, dataset, w0_dict, batch_size=200, num_iter=10, no_bp=False):
    """Compute the full information with back propagation support.
    Using delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2 for efficient computation.
    Args:
        no_bp: detach the information term hence it won't be used for learning.
    """
    param_keys = [p[0] for p in model.named_parameters()]
    delta_w_dict = dict().fromkeys(param_keys)
    for pa in model.named_parameters():
        if "weight" in pa[0]:
            w0 = w0_dict[pa[0]]
            delta_w = pa[1] - w0
            delta_w_dict[pa[0]] = delta_w

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    info_dict = dict()
    gw_dict = dict().fromkeys(param_keys)

    for it, (images, target) in enumerate(loader):
        if it >= num_iter:
            break
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        pred = model.forward(images)
        loss = F.cross_entropy(pred, target, reduction="mean")

        gradients = grad(loss, model.parameters())

        for i, gw in enumerate(gradients):
            gw_ = gw.flatten()
            if gw_dict[param_keys[i]] is None:
                gw_dict[param_keys[i]] = gw_
            else:
                gw_dict[param_keys[i]] += gw_

    num_all_batch = int(np.ceil(len(dataset) / batch_size))
    for k in gw_dict.keys():
        if "weight" in k:
            gw_dict[k] *= 1 / num_all_batch
            delta_w = delta_w_dict[k]
            # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
            info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
            if no_bp:
                info_dict[k] = info_.item()
            else:
                info_dict[k] = info_

    return info_dict
