import argparse
import itertools
import os
import queue
from time import sleep

from dataset import get_num_samples
from utils import Task, wait_gpu


def main(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    todo_queue = queue.Queue()
    running_list = list()
    print("=> submitting tasks")

    for seed, s_seed, config, ns in itertools.product(seeds, s_seeds,
                                                      configs,
                                                      num_samples):
        task_name = f"{dataset}-{args.ghost_dataset_name}_{config['method']}_SEED{seed}_SSEED{s_seed}_NS{ns}"
        if args.beta is None:
            if dataset == 'mnist':
                beta, lam = 10000.0, 10.0
            elif dataset == 'fashion':
                beta, lam = 100.0, 10.0
            elif num_samples.index(ns) == 0:
                beta, lam = 100.0, 10.0
            elif num_samples.index(ns) == 1:
                beta, lam = 10.0, 10.0
            elif dataset == 'svhn':
                beta, lam = 10.0, 10.0
            else:
                beta, lam = 1.0, 1.0
        else:
            beta, lam = args.beta, 10.0

        if os.path.isfile(os.path.join(args.logdir, dataset, task_name, 'results.json')):
            continue
        cmd = \
            f'''
        {args.interpreter} train.py {dataset} \
      --datadir={args.datadir} \
      --logdir={args.logdir}/{dataset}/{task_name} \
      --arch={arch} \
      --cri_arch={cri_arch} \
      --epoch={config['epoch']} \
      --batch-size={config['bs']} \
      --lr={config['lr']} \
      --wd={config['wd']} \
      --print-freq=100 \
      --beta={beta} \
      --lam={lam} \
      --lr-policy=cosine \
      --dataset_on_gpu \
      --warmup-epochs={config['warm']} \
      --num-hiddens={config['nh']} \
      --dropout={config['dropout']} \
      --seed={seed} \
      --S-seed={s_seed} \
      --num-samples={ns} \
      --method={config['method']} \
      --save_freq={args.save_freq} \
      --early_stop_tolerance={config['es']} \
      --error_prob={args.error_prob} 
        '''
        if args.ghost_dataset_name is not None:
            cmd += f"--ghost_dataset_name={args.ghost_dataset_name}"
        cmd = cmd.split()
        task = Task(task_name, cmd, args.logdir)
        todo_queue.put(task)
    print("=> done!")

    gpus = []
    while not todo_queue.empty():
        task = todo_queue.get()
        reserve_list = args.reserve_gpus if args.reserve_gpus is not None else []
        gpus = wait_gpu(num=1, usage_threshold=args.usage_threshold,
                        waitsecs=args.waitsecs, reserve_gpus=reserve_list + gpus)
        task.start(gpus)
        print("=> run task {}".format(task))
        running_list.append(task)
        print("=> cold down {} seconds".format(args.coldsecs))
        sleep(args.coldsecs)

    print("=> all tasks submitted, waiting for finish...")
    for task in running_list:
        duration = task.wait()
        print("=> finish task '{}' in {}".format(task.name, duration))
    print("=> all tasks finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coldsecs', type=int, default=12)
    parser.add_argument('--waitsecs', type=int, default=30)
    parser.add_argument('--usage_threshold', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--reserve_gpus', nargs='+', type=int, default=None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--tag', type=str, default='exp')
    parser.add_argument('--logdir', type=str, default='.')
    parser.add_argument('--datadir', type=str, default='.')
    parser.add_argument('--arch', type=str, default='my_cnn.CNN')
    parser.add_argument('--cri_arch', type=str, default='my_mlp.MLP')
    parser.add_argument('--save_freq', type=int, default=-1)
    parser.add_argument('--ghost_dataset_name', type=str, default=None)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--num_S_seeds', type=int, default=5)
    parser.add_argument('--interpreter', type=str, default='python')
    parser.add_argument('--error_prob', type=float, default=0.0)

    args = parser.parse_args()
    num_seeds = args.num_seeds
    num_S_seeds = args.num_S_seeds
    seeds = range(num_seeds)
    s_seeds = range(num_S_seeds)
    dataset = args.dataset
    num_samples = get_num_samples(dataset)

    arch = args.arch
    cri_arch = args.cri_arch
    base_cfg = {'warm': 0, 'epoch': 100, 'bs': 128, 'lr': 1e-3,
                'wd': 0.0, 'dropout': 0.0, 'es': 20, 'nh': 3}
    vanilla_cfg = {**base_cfg, 'method': 'vanilla'}
    l2_cfg = {**base_cfg, 'method': 'l2', 'wd': 1e-4}
    dropout_cfg = {**base_cfg, 'method': 'dropout', 'dropout': 0.1}
    pib_cfg = {**base_cfg, 'method': 'pib', 'es': 100}
    vib_cfg = {**base_cfg, 'method': 'vib', 'es': 100}
    nib_cfg = {**base_cfg, 'method': 'nib', 'es': 100}
    dib_cfg = {**base_cfg, 'method': 'dib', 'es': 100}
    rib_cfg = {**base_cfg, 'method': 'rib', 'es': 100}
    rib_minimax_cfg = {**base_cfg, 'method': 'rib_minimax', 'es': 100}
    rib_sq_cfg = {**base_cfg, 'method': 'rib_sq', 'es': 100}
    rib_ukl_cfg = {**base_cfg, 'method': 'rib_ukl', 'es': 100}
    configs = [vanilla_cfg, l2_cfg, dropout_cfg, pib_cfg, vib_cfg, nib_cfg, dib_cfg, 
               rib_cfg, rib_minimax_cfg, rib_sq_cfg, rib_ukl_cfg]
    if args.method is not None:
        configs = [c for c in configs if c['method'] == args.method]
    args.logdir = os.path.join(args.logdir, f'{arch}_{cri_arch}_seed_{num_seeds}_{num_S_seeds}_{args.tag}')

    main(args)
