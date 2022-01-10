import sys


from fedlab.utils.functional import save_dict
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

import torchvision
num_shards = 200
num_client=10
trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=False)

data_indices = noniid_slicing(trainset, num_clients=num_client, num_shards=num_shards)
save_dict(data_indices, "cifar10_non_client_{}_shards_{}.pkl".format(num_client, num_shards))

data_indices = random_slicing(trainset, num_clients=num_client)
save_dict(data_indices, "cifar10_iid_client_{}_shards_{}.pkl".format(num_client, num_shards))

"""
Please refer to cifar10_partition.ipynb file for usage of CIFAR10Partitioner.

Function ``random_slicing()`` and ``noniid_slicing()`` are deprecated in the future version.
"""
