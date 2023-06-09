import argparse
import os.path

import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR',
                    help='path to datasets')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--new-size', type=int, default=512)
parser.add_argument('--crop-size', type=int, default=448)
parser.add_argument('--datadir', type=str, default='.')
parser.add_argument('--logdir', type=str, default='.')
parser.add_argument('--warmup-epochs', type=int, default=0)
parser.add_argument('--lr-step', type=int, default=None)
parser.add_argument('--milestones', nargs='+', type=int, default=None)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--lr-policy', type=str)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--num-hiddens', type=int)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--S-seed', type=int, required=True)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--dataset_on_gpu', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--cri_arch', type=str)
parser.add_argument('--num-samples', type=int, required=True)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--aug_train', action='store_true')
parser.add_argument('--method', type=str, choices=['rib', 'vanilla', 'l2', 'dropout', 'pib', 'vib', 'nib', 'dib', 'rib_minimax', 'rib_sq', 'rib_ukl'])
parser.add_argument('--early_stop_tolerance', type=int, default=-1)
parser.add_argument('--save_freq', default=-1, type=int)
parser.add_argument('--ghost_dataset_name', type=str, default=None)
parser.add_argument('--error_prob', type=float, default=0.0)


def main():
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        free_gpus = get_free_gpu(num=1)
        os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
    os.environ["OMP_NUM_THREADS"] = str(2)

    if not args.evaluate:
        save_current_code(args, __file__)
    set_random_seed(args.seed)

    train_dataset, ghost_dataset, _, val_dataset, test_dataset, _ = \
        get_all_datasets(args.data,
                         num_samples=args.num_samples,
                         seed=args.seed,
                         S_seed=args.S_seed,
                         gpu=args.dataset_on_gpu,
                         root=args.datadir,
                         ghost_dataset_name=args.ghost_dataset_name,
                         error_prob=args.error_prob)
    writer = SummaryWriter(args.logdir)
    if args.resume:
        with open(os.path.join(args.logdir, 'results.json'), 'r') as f:
            result_dict = json.load(f)
    else:
        result_dict = {}

    print("=> baseline training")
    main_baseline(args, writer, result_dict, train_dataset, ghost_dataset, val_dataset, test_dataset)

    save_result_dict(result_dict, args.logdir, filename='results.json')
    print("=> finished")


def main_baseline(args, writer, result_dict, train_dataset, ghost_dataset, val_dataset, test_dataset):
    num_classes = args.num_classes = get_dataset_class_number(args.data)

    print("=> creating model '{}'".format(args.arch))
    model = get_network(args.arch, num_classes=num_classes, dropout_rate=args.dropout,
                        input_channels=1 if args.data in ('mnist', 'fashion') else 3,
                        reparametrize=args.method)

    model.cuda()
    print(model)

    if args.method == 'pib':
        w0_dict = dict()
        for param in model.named_parameters():
            w0_dict[param[0]] = param[1].clone().detach()  # detach but still on gpu
        model = get_network(args.arch, num_classes=num_classes, dropout_rate=args.dropout,
                            input_channels=1 if args.data in ('mnist', 'fashion') else 3,
                            reparametrize=args.method)
        model.cuda()
    
    if args.resume:
        print("=> loading checkpoint from '{}'".format(args.logdir))
        state_dict = torch.load(os.path.join(args.logdir, "checkpoint.pth"), map_location='cuda:0')
        model.load_state_dict(state_dict['state_dict'])
        return model
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True,
                              pin_memory=not args.dataset_on_gpu)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, shuffle=False,
                            pin_memory=not args.dataset_on_gpu)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size, shuffle=False,
                             pin_memory=not args.dataset_on_gpu)

    scheduler = get_scheduler(args, optimizer, T_max=len(train_loader) * args.epochs)
    
    extra_params = dict()
    if args.method.startswith('rib'):
        ghost_loader = DataLoader(ghost_dataset,
                                  batch_size=args.batch_size, shuffle=True,
                                  pin_memory=not args.dataset_on_gpu)

        model_cri = get_network(args.cri_arch,
                                in_features=model.feat_size * 2,
                                num_hiddens=args.num_hiddens)
        model_cri.cuda()
        optimizer_cri = torch.optim.SGD(model_cri.parameters(), args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        scheduler_cri = get_scheduler(args, optimizer_cri, T_max=len(ghost_loader) * args.epochs)
        bregman_type = 'bkl'
        if args.method == 'rib_minimax':
            train_fn = train_rib_minimax
        else:
            train_fn = train_rib
            if args.method == 'rib_sq':
                bregman_type = 'sq'
            elif args.method == 'rib_ukl':
                bregman_type = 'ukl'
                

        extra_params = dict(ghost_loader=ghost_loader, model_cri=model_cri, optimizer_cri=optimizer_cri,
                            scheduler_cri=scheduler_cri, bregman_type=bregman_type)
    elif args.method == 'pib':
        train_fn = train_pib
        extra_params = dict(energy_decay=torch.zeros(1))
    elif args.method == 'vib':
        train_fn = train_vib
    elif args.method == 'nib':
        train_fn = train_nib
    elif args.method == 'dib':
        train_fn = train_dib
    else:
        train_fn = train_baseline

    train_acc1, val_acc1, test_acc1, best_acc1 = 0.0, 0.0, 0.0, 0.0
    early_stop_counter = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_acc1 = train_fn(train_loader=train_loader, model=model, optimizer=optimizer, scheduler=scheduler,
                              args=args, epoch=epoch, writer=writer, **extra_params)
        if args.method == 'pib':
            info = compute_information_bp_fast(model, train_dataset, w0_dict)
            energy_decay = 0
            for k in info.keys():
                energy_decay += info[k]
            energy_decay = 0.1 * energy_decay
            extra_params['energy_decay'] = energy_decay

        # evaluate on validation set
        val_acc1 = validate(val_loader, model, args, epoch, writer, tag='val')
        test_acc1 = validate(test_loader, model, args, epoch, writer, tag='test')

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        if is_best:
            early_stop_counter = 0
            best_acc1 = val_acc1
        else:
            early_stop_counter += 1
        print("=> LR = {}".format(scheduler.get_last_lr()[0]))
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('GenBound/val', train_acc1 - val_acc1, epoch)
        writer.add_scalar('GenBound/test', train_acc1 - test_acc1, epoch)
        if args.early_stop_tolerance > 0 and early_stop_counter >= args.early_stop_tolerance:
            print("early stop on epoch {}, val acc {}".format(epoch, best_acc1))
            break
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            state_dict = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'best_acc1': best_acc1}
            save_checkpoint(state_dict, filename=f'checkpoint_{epoch}.pth', logdir=args.logdir)
            result = {'test_acc': test_acc1,
                      'val_acc': val_acc1,
                      'train_acc': train_acc1,
                      'val_bound': train_acc1 - val_acc1,
                      'test_bound': train_acc1 - test_acc1}
            save_result_dict(result, args.logdir, filename=f'results_{epoch}.json')

    print("=> saving results")
    final_state_dict = {'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1}
    save_checkpoint(final_state_dict, logdir=args.logdir)

    result = {'test_acc': test_acc1,
              'val_acc': val_acc1,
              'best_val_acc': best_acc1,
              'train_acc': train_acc1,
              'val_bound': train_acc1 - val_acc1,
              'test_bound': train_acc1 - test_acc1}
    result_dict |= result
    return model


if __name__ == '__main__':
    main()
