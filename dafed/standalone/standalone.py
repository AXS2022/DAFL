import time
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
import shutil

torch.manual_seed(0)

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import get_best_gpu, load_dict, AverageMeter

from dafed.local.localclient import LocalTrainer

sys.path.append("../../../")
from dafed.local.cnn import CNN_MNIST
from dafed.local.mlp import MLP
from dafed.local.cnn import CNN_FMNIST
from dafed.local.resnet import resnet20




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
parser.add_argument("--com_round", type=int, default=400)

parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int,default=16)
parser.add_argument("--dataset", type=str,default="cifar10")
parser.add_argument("--model",type=str,default="cnn")
parser.add_argument("--partition", type=str, default='noniid')
parser.add_argument("--shards", type=str, default='300')
parser.add_argument("--file_id", type=str, default='5')

args = parser.parse_args()


print("init arg:{}".format(args))

# get raw dataset

# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

gpu = get_best_gpu()
model = None
trainset = None
testset=None
test_loader = None
transform = transforms.Compose(
    [
     transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))])


timestamp = time.time()
localtime= time.localtime(timestamp)
daytime = time.strftime("%Y_%m_%d_%H_%M_%S", localtime)


def evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy."""
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.avg




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
    if args.model == "mlp":
        model = MLP(784, 10)
    if args.model == "cnn":
        model = CNN_MNIST()

if args.dataset=="cifar10":

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
        model = resnet20()


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
# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client  # client??????

data_indices  = []
if args.dataset == "mnist":
    if args.partition == "noniid":
        data_indices = load_dict("mnist_noniid_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
    else:
        data_indices = load_dict("mnist_iid_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
if args.dataset == "fmnist":
        if args.partition == "noniid":
            data_indices = load_dict("fmnist_noniid_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
        else:
            data_indices = load_dict("fmnist_iid_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
if args.dataset == "cifar10":
    if args.partition == "noniid":
        data_indices = load_dict("cifar10_non_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
    else:
        data_indices = load_dict("cifar10_iid_client_{}_shards_{}.pkl".format(args.total_client, args.shards))
# fedlab setup

local_model = deepcopy(model)
save_model_dir = "../save_local_model_{}".format(args.file_id)
if os.path.isdir(save_model_dir) is True:
    shutil.rmtree(save_model_dir)
    os.mkdir(save_model_dir)
else:
    os.mkdir(save_model_dir)
trainer = LocalTrainer(model=local_model,
                       dataset=trainset,
                       data_slices=data_indices,
                       aggregator=aggregator,
                       cuda=True,
                       args={
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "file_id": args.file_id,
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

    print("round:{}-th, loss: {:.4f}, acc: {:.4f}".format(round,loss, acc))
    total_loss.append(loss)
    total_acc.append(acc)


file_name = "{}_{}_{}_{}_{}_shards_{}data.txt".format(args.file_id, daytime, args.dataset, args.partition, args.model,args.shards)
with open(file_name,'w') as f:
    f.write("loss:\n{}\n acc:\n{}".format(total_loss, total_acc))
