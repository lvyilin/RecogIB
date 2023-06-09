import os.path

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, FashionMNIST, STL10

_dataset_fn = {
    'mnist': MNIST,
    'fashion': FashionMNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'svhn': SVHN,
    'stl10': STL10
}
_dataset_class_number_map = {
    'mnist': 10,
    'fashion': 10,
    'cifar10': 10,
    'cifar100': 100,
    'svhn': 10,
    'stl10': 10,
}


def to_tensor_dataset(dataset, gpu=False):
    xs, ys = [], []
    for x, y in dataset:
        xs.append(x)
        ys.append(y)
    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys)
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return torch.utils.data.TensorDataset(xs, ys)


def _get_dataset(dataset_name, **kwargs):
    if dataset_name in ('svhn', 'stl10') and 'split' not in kwargs:
        split = 'train' if kwargs['train'] else 'test'
        kwargs['split'] = split
        del kwargs['train']
    return _dataset_fn[dataset_name](**kwargs)


def get_dataset_class_number(dataset_name):
    return _dataset_class_number_map[dataset_name]


def get_transform(data):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    if data == 'stl10':
        transforms_list.append(transforms.Resize(32))
    if data not in ('mnist', 'fashion'):
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))

    return transforms.Compose(transforms_list)


def get_all_datasets(dataset_name, root, num_samples, seed, S_seed, gpu=False,
                     ghost_dataset_name=None, error_prob=0.0):
    orig_train_dataset = _get_dataset(dataset_name, root=os.path.join(root, dataset_name), train=True,
                                      transform=get_transform(dataset_name))
    if ghost_dataset_name is not None:
        orig_ghost_dataset = _get_dataset(ghost_dataset_name, root=os.path.join(root, ghost_dataset_name), train=True,
                                          transform=get_transform(ghost_dataset_name))
        if len(orig_ghost_dataset) < len(orig_train_dataset):
            r = np.random.RandomState(seed)
            include_indices = r.choice(len(orig_ghost_dataset), size=len(orig_train_dataset), replace=True)
            orig_ghost_dataset = torch.utils.data.Subset(orig_ghost_dataset, include_indices)

    else:
        orig_ghost_dataset = orig_train_dataset
    test_dataset = _get_dataset(dataset_name, root=os.path.join(root, dataset_name), train=False,
                                transform=get_transform(dataset_name))

    orig_train_dataset = to_tensor_dataset(orig_train_dataset, gpu)
    if ghost_dataset_name is not None:
        orig_ghost_dataset = to_tensor_dataset(orig_ghost_dataset, gpu)

    test_dataset = to_tensor_dataset(test_dataset, gpu)

    r = np.random.RandomState(seed)
    include_indices = r.choice(len(orig_train_dataset), size=2 * num_samples, replace=False)
    r = np.random.RandomState(S_seed)
    mask = r.randint(2, size=(num_samples,))
    train_indices = include_indices[2 * np.arange(num_samples) + mask]
    val_indices = include_indices[2 * np.arange(num_samples) + (1 - mask)]

    train_dataset = torch.utils.data.Subset(orig_train_dataset, train_indices)        
    val_dataset = torch.utils.data.Subset(orig_train_dataset, val_indices)
    train_indices_mask = np.zeros(len(orig_train_dataset), dtype=bool)
    train_indices_mask[train_indices] = 1
    exclude_indices = np.where(~train_indices_mask)[0]
    ghost_indices = r.choice(len(exclude_indices), size=num_samples, replace=False)
    ghost_dataset = torch.utils.data.Subset(orig_ghost_dataset, exclude_indices[ghost_indices])
    # ghost_dataset = torch.utils.data.Subset(orig_ghost_dataset, val_indices)
    if ghost_dataset_name is not None:
        sup_dataset = Supersample(train_dataset, ghost_dataset, mask)
    else:
        sup_dataset = torch.utils.data.Subset(orig_train_dataset, include_indices)
    if error_prob > 0.0:
        print(f"=> Set noisy dataset with error prob: {error_prob}")
        train_dataset = NoisyDataset(train_dataset, error_prob, get_dataset_class_number(dataset_name), r)
        
    print(f'#Train: {len(train_dataset)}, #Ghost: {len(ghost_dataset)}, #Super: {len(sup_dataset)}, '
          f'#Val: {len(val_dataset)}, #Test: {len(test_dataset)}')
    return train_dataset, ghost_dataset, sup_dataset, val_dataset, test_dataset, mask

class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, error_prob, num_classes, random_state=None):
        self.dataset = dataset
        self.error_prob = error_prob
        self.num_classes =num_classes
        
        confusion_matrix = error_prob / (num_classes - 1) * np.ones((num_classes, num_classes))
        for i in range(num_classes):
            confusion_matrix[i, i] = 1 - error_prob
            
        self.is_corrupted = np.zeros(len(dataset), dtype=np.bool)  # 0 clean, 1 corrupted
        self.new_targets = []
        
        if random_state is None:
            random_state = np.random
        for i, (_, label) in enumerate(dataset):
            new_label = int(random_state.choice(num_classes, 1, p=np.array(confusion_matrix[label])))
            self.is_corrupted[i] = (int(label) != new_label)
            self.new_targets.append(new_label)
    def __getitem__(self, index):
        return self.dataset[index][0], self.new_targets[index]
    def __len__(self):
        return len(self.dataset)


class Supersample(torch.utils.data.Dataset):
    def __init__(self, train_dataset, ghost_dataset, mask):
        assert len(train_dataset) == len(ghost_dataset)
        self.train_dataset = train_dataset
        self.ghost_dataset = ghost_dataset
        self.mask = mask

    def __getitem__(self, index):
        if self.mask[index // 2] == 0:
            if index % 2 == 0:
                return self.train_dataset[index // 2]
            return self.ghost_dataset[index // 2]
        if index % 2 == 0:
            return self.ghost_dataset[index // 2]
        return self.train_dataset[index // 2]

    def __len__(self):
        return len(self.train_dataset) + len(self.ghost_dataset)


def get_num_samples(dataset):
    if dataset == 'mnist':
        num_samples = [1250, 5000, 20000]
    elif dataset == 'fashion':
        num_samples = [1250, 5000, 20000]
    elif dataset == 'cifar10':
        num_samples = [1250, 5000, 20000]
    elif dataset == 'svhn':
        num_samples = [1250, 5000, 20000]
    elif dataset == 'stl10':
        num_samples = [625, 1250, 2500]
    else:
        raise NotImplementedError
    return num_samples
