import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dis_in, dis_hidden):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(dis_in, dis_hidden),
            nn.LeakyReLU(0.2),
        )
        self.hid_layer = nn.Sequential(
            nn.Linear(dis_hidden, dis_hidden),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Linear(dis_hidden,1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hid_layer(x)
        x = self.out(x)
        x = self.sigmod(x)

        return x
