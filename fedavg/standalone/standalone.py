from json import load
import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import sys
import torch
import time

torch.manual_seed(0)

from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu, load_dict

sys.path.append("../../../")

from models.cnn import CNN_MNIST
from models.cnn import CNN_FMNIST
from models.mlp import MLP
from models.cnn import AlexNet_CIFAR10,CNN_cifar10


def write_file(acc, loss, config, round):
    record = open(
        "{}_{}_{}_{}.txt".format(config.partition, config.sample_ratio,
                                 config.batch_size, config.epochs), "w")
    record.write(str(round) + "\n")
    record.write(str(config) + "\n")
    record.write(str(loss) + "\n")
    record.write(str(acc) + "\n")
    record.close()


# python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition iid

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int, default=200)

parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int,default=16)
parser.add_argument("--dataset", type=str,default="cifar10")
parser.add_argument("--model",type=str,default="cnn")
parser.add_argument("--partition", type=str, default='noniid')
parser.add_argument("--shards", type=str, default='200')

args = parser.parse_args()
print("init arg:{}".format(args))

# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# gpu = get_best_gpu()
model = None
trainset = None
testset=None
test_loader = None
transform = transforms.Compose(
    [
     transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))])
# get raw dataset
if args.dataset=="mnist":
    root = "../../datasets/mnist/"
    trainset = torchvision.datasets.MNIST(root=root,
                                          train=True,
                                          download=True,
                                          transform=transform)

    testset = torchvision.datasets.MNIST(root=root,
                                         train=False,
                                         download=True,
                                         transform=transform)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=len(testset),
                                              drop_last=False,
                                              shuffle=False)
    if args.model=="mlp":
        model = MLP(784,10)
    if args.model=="cnn":
        model = CNN_MNIST()
elif args.dataset=="cifar10":
    root = "../../datasets/cifar10"
    trainset = torchvision.datasets.CIFAR10(root=root,
                                          train=True,
                                          download=True,
                                          transform=transform)

    testset = torchvision.datasets.CIFAR10(root=root,
                                         train=False,
                                         download=True,
                                         transform=transform)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=len(testset),
                                              drop_last=False,
                                              shuffle=False)
    if args.model=="mlp":
        model = MLP(1024*3,10)
    if args.model=="cnn":
        model = CNN_cifar10()


if args.dataset=="fmnist":
    root = "../../datasets/fmnist/"
    trainset = torchvision.datasets.FashionMNIST(root=root,
                                          train=True,
                                          download=True,
                                          transform=transform)

    testset = torchvision.datasets.FashionMNIST(root=root,
                                         train=False,
                                         download=True,
                                         transform=transform)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=len(testset),
                                              drop_last=False,
                                              shuffle=False)
    if args.model == "mlp":
        model = MLP(784, 10)
    if args.model == "cnn":
        model =CNN_FMNIST()





#model = CNN_MNIST().cuda(gpu)


# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client  # client总数

data_indices  = []
if args.dataset == "mnist":
    if args.partition == "noniid":
        data_indices = load_dict("mnist_noniid_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
    else:
        data_indices = load_dict("mnist_iid_client_{}_shards_{}.pkl")
if args.dataset == "fmnist":
        if args.partition == "noniid":
            data_indices = load_dict("fmnist_noniid_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
        else:
            data_indices = load_dict("fmnist_iid_client_{}_shards_{}.pkl")
if args.dataset == "cifar10":
    if args.partition == "noniid":
        data_indices = load_dict("cifar10_non_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
    else:
        data_indices = load_dict("cifar10_iid_client_{}_shards_{}.pkl")
print(len(data_indices))
# fedlab setup
local_model = deepcopy(model)

trainer = SubsetSerialTrainer(model=local_model,
                              dataset=trainset,
                              data_slices=data_indices,
                              aggregator=aggregator,
                              cuda=False,
                              args={
                                  "batch_size": args.batch_size,
                                  "epochs": args.epochs,
                                  "lr": args.lr
                              })

# train procedure
to_select = [i for i in range(total_client_num)]
total_acc = []
total_loss = []

for round in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    selection = random.sample(to_select, num_per_round)
    aggregated_parameters = trainer.train(model_parameters=model_parameters,
                                          id_list=selection,
                                          aggregate=True)

    SerializationTool.deserialize_model(model, aggregated_parameters)

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print("round: {}, loss: {:.4f}, acc: {:.4f}".format(round,loss, acc))
    total_loss.append(loss)
    total_acc.append(acc)

timestamp = time.time()
localtime= time.localtime(timestamp)
daytime = time.strftime("%Y_%m_%d_%H_%M_%S", localtime)

file_name = "{}_{}_{}_{}_shards_{}data.txt".format(daytime, args.dataset, args.partition, args.model,args.shards)
print(total_acc)
with open(file_name,'w') as f:
    f.write("loss:\n{}\n acc:\n{}".format(total_loss, total_acc))