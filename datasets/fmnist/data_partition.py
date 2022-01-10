import sys


from fedlab.utils.functional import save_dict
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

import torchvision
num_shards = 300
num_client=100
trainset = torchvision.datasets.FashionMNIST(root="./", train=True, download=False)

data_indices = noniid_slicing(trainset, num_clients=num_client, num_shards=num_shards)
save_dict(data_indices, "fmnist_noniid_client_{}_shards_{}.pkl".format(num_client, num_shards))

data_indices = random_slicing(trainset, num_clients=num_shards)
save_dict(data_indices, "fmnist_iid_client_{}_shards_{}.pkl".format(num_client, num_shards))