#!/usr/bin/env bash
# Train model
python run_all.py --dataset=fashion --interpreter=$1 --logdir=$2 --datadir=$3
python run_all.py --dataset=svhn --interpreter=$1 --logdir=$2 --datadir=$3
python run_all.py --dataset=cifar10 --interpreter=$1 --logdir=$2 --datadir=$3
python run_all.py --dataset=stl10 --interpreter=$1 --logdir=$2 --datadir=$3
python run_all.py --dataset=mnist --interpreter=$1 --logdir=$2 --datadir=$3
# Estimate recognizability
python run_all_recog.py --dataset=fashion --interpreter=$1 --logdir=$2 --datadir=$3
python run_all_recog.py --dataset=svhn --interpreter=$1 --logdir=$2 --datadir=$3
python run_all_recog.py --dataset=cifar10 --interpreter=$1 --logdir=$2 --datadir=$3
python run_all_recog.py --dataset=stl10 --interpreter=$1 --logdir=$2 --datadir=$3
python run_all_recog.py --dataset=mnist --interpreter=$1 --logdir=$2 --datadir=$3
# Change sources
python run_all.py --dataset=cifar10 --ghost_dataset_name=svhn --method=rib --interpreter=$1 --logdir=$2 --datadir=$3
python run_all.py --dataset=cifar10 --ghost_dataset_name=cifar100 --method=rib --interpreter=$1 --logdir=$2 --datadir=$3
python run_all_recog.py --dataset=cifar10 --ghost_dataset_name=svhn --method=rib --interpreter=$1 --logdir=$2 --datadir=$3
python run_all_recog.py --dataset=cifar10 --ghost_dataset_name=cifar100 --method=rib --interpreter=$1 --logdir=$2 --datadir=$3

# Change beta
python run_tune.py  --tag=exp_tune --dataset=fashion --interpreter=$1 --logdir=$2 --datadir=$3 --method=nib
python run_tune.py  --tag=exp_tune --dataset=svhn --interpreter=$1 --logdir=$2 --datadir=$3 --method=nib
python run_tune.py  --tag=exp_tune --dataset=cifar10 --interpreter=$1 --logdir=$2 --datadir=$3 --method=nib
python run_tune.py  --tag=exp_tune --dataset=stl10 --interpreter=$1 --logdir=$2 --datadir=$3 --method=nib
python run_tune.py  --tag=exp_tune --dataset=mnist --interpreter=$1 --logdir=$2 --datadir=$3 --method=nib
python run_tune.py  --tag=exp_tune --dataset=fashion --interpreter=$1 --logdir=$2 --datadir=$3 --method=dib
python run_tune.py  --tag=exp_tune --dataset=svhn --interpreter=$1 --logdir=$2 --datadir=$3 --method=dib
python run_tune.py  --tag=exp_tune --dataset=cifar10 --interpreter=$1 --logdir=$2 --datadir=$3 --method=dib
python run_tune.py  --tag=exp_tune --dataset=stl10 --interpreter=$1 --logdir=$2 --datadir=$3 --method=dib
python run_tune.py  --tag=exp_tune --dataset=mnist --interpreter=$1 --logdir=$2 --datadir=$3 --method=dib

# Plot recog dynamics
python run_all.py --dataset=cifar10 --method=vanilla --interpreter=$1 --logdir=$2 --datadir=$3 --save_freq=5  --tag=plot_dynamic
python run_all_recog_epochwise.py --dataset=cifar10 --method=vanilla --interpreter=$1 --logdir=$2 --datadir=$3 --save_freq=5  --tag=plot_dynamic