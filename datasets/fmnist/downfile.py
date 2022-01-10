import torchvision

if __name__ == "__main__":
    root = './'
    trainset = torchvision.datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
    )

    testset = torchvision.datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
    )
