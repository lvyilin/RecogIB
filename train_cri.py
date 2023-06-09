import argparse
import os.path

import torch.optim
from torch.utils.data import DataLoader

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
parser.add_argument('--num-hiddens', type=int)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--S-seed', type=int, required=True)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--dataset_on_gpu', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--cri_arch', type=str)
parser.add_argument('--num-samples', type=int, required=True)
parser.add_argument('--num_training_samples', type=int, required=True)
parser.add_argument('--aug_train', action='store_true')
parser.add_argument('--method', type=str, choices=['rib', 'vanilla', 'l2', 'dropout', 'pib', 'vib'])
parser.add_argument('--ghost_dataset_name', type=str, default=None)


def main():
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        free_gpus = get_free_gpu(num=1)
        os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
    os.environ["OMP_NUM_THREADS"] = str(2)

    set_random_seed(args.seed)

    train_dataset, ghost_dataset, _, val_dataset, test_dataset, _ = \
        get_all_datasets(args.data,
                         num_samples=args.num_training_samples,
                         seed=args.seed,
                         S_seed=args.S_seed,
                         gpu=args.dataset_on_gpu,
                         root=args.datadir,
                         ghost_dataset_name=args.ghost_dataset_name)

    model = get_trained_network(args, args.arch, args.logdir)

    with open(os.path.join(args.logdir, 'results.json'), 'r') as f:
        result_dict = json.load(f)

    writer = None  # SummaryWriter(args.logdir)

    print(f"=> critic training")
    model_critic = main_critic(args, writer, model, train_dataset, ghost_dataset)
    print(f"=> critic evaluating")
    _, _, sup_dataset, _, _, mask = \
        get_all_datasets(args.data,
                         num_samples=args.num_samples,
                         seed=args.seed,
                         S_seed=args.S_seed,
                         gpu=args.dataset_on_gpu,
                         root=args.datadir,
                         ghost_dataset_name=args.ghost_dataset_name)
    assert 2 * len(mask) == len(sup_dataset)
    sup_loader = DataLoader(sup_dataset,
                            batch_size=args.batch_size * 2, shuffle=False,
                            pin_memory=not args.dataset_on_gpu)

    recog_est = estimate_recog(sup_loader, mask, model, model_critic, args)
    save_prediction({'pred': get_prediction(sup_loader, model), 'mask': mask, 'recog': recog_est}, 'preds.pkl',
                    args.logdir)

    result_filename = 'results_recog.json'
    save_result_dict(result_dict, args.logdir, filename=result_filename)
    print("=> finished")


def main_critic(args, writer, model, train_dataset, ghost_dataset):
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True,
                              pin_memory=not args.dataset_on_gpu)
    ghost_loader = DataLoader(ghost_dataset,
                              batch_size=args.batch_size, shuffle=True,
                              pin_memory=not args.dataset_on_gpu)

    print("=> creating model '{}'".format(args.cri_arch))
    model_cri = get_network(args.cri_arch,
                            in_features=model.feat_size * 2,
                            num_hiddens=args.num_hiddens)
    model_cri.cuda()

    optimizer_cri = torch.optim.SGD(model_cri.parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scheduler = get_scheduler(args, optimizer_cri, T_max=len(ghost_loader) * args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        train_critic(train_loader, ghost_loader, model, model_cri, optimizer_cri, scheduler, args, epoch, writer=writer)
    return model_cri


def get_trained_network(args, arch, ckpt_dir, epoch=None):
    num_classes = get_dataset_class_number(args.data)

    print("=> creating model '{}'".format(arch))
    model = get_network(arch,
                        num_classes=num_classes, dropout_rate=args.dropout,
                        input_channels=1 if args.data in ('mnist', 'fashion') else 3,
                        reparametrize=args.method)
    print("=> loading checkpoint from '{}'".format(args.logdir))
    filename = f"checkpoint{'' if epoch is None else '_' + str(epoch)}.pth"
    state_dict = torch.load(os.path.join(ckpt_dir, filename), map_location='cuda:0')
    model.load_state_dict(state_dict['state_dict'])
    model.cuda()
    return model


if __name__ == '__main__':
    main()
