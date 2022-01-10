import os
import torch
import torch.nn.functional as F
from fedlab.core.client import SerialTrainer
from fedlab.utils import SerializationTool
from fedlab.utils.dataset import SubsetSampler
from fedlab.utils.functional import save_dict, load_dict
from fedlab.utils.logger import Logger

client_buffer = {}


class LocalTrainer(SerialTrainer):

    def __init__(self, model, dataset, data_slices, aggregator, logger=Logger(), cuda=True, args=None):
        super(LocalTrainer, self).__init__(model=model,
                                           client_num=len(data_slices),
                                           cuda=cuda,
                                           aggregator=aggregator,
                                           logger=logger
                                           )
        self.dataset = dataset
        self.data_slices = data_slices  # [0, client_num)
        self.args = args
        self.logger = logger
        self.local_model = None
    def _get_dataloader(self, client_id):

        batch_size = self.args["batch_size"]
        self.client_id = client_id
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[client_id],
                                  shuffle=True),
            batch_size=batch_size)

        return train_loader

    def _get_local_model(self):
       save_file = "../save_local_model_{}/{}.pkl".format(self.args["file_id"], self.client_id)
       print(save_file)
       if os.path.isfile(save_file):
           self.local_model.load_state_dict(load_dict(save_file))
           self.local_model.eval()
       else:
            self.local_model = self.model

    def _train_alone(self, model_parameters, train_loader):

        """Single round of local training for one client.

               Note:
                   Overwrite this method to customize the PyTorch training pipeline.

               Args:
                   model_parameters (torch.Tensor): serialized model parameters.
                   train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
               """
        self._get_local_model()
        epochs, lr = self.args["epochs"], self.args["lr"]
        SerializationTool.deserialize_model(self.model, model_parameters)

        criterion = torch.nn.CrossEntropyLoss()
        KLcriterion  = torch.nn.KLDivLoss(reduction="batchmean")
        BEcriterion =  torch.nn.BCELoss(reduction="mean")
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr, momentum=0.5)
        local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.5)
        self._model.train()
        self.local_model.train()

        a = 0
        b = 0
        total = 0
        for ep in range(epochs):

            for data, target in train_loader:
                #print(target)
                one_label = torch.ones((len(target),1))
                zero_label = torch.zeros((len(target),1))
                total += len(target)
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)
                    one_label = one_label.cuda(self.gpu)
                    zero_label = zero_label.cuda(self.gpu)
                output = self.model(data)
                self.local_model(data)
                n1 = torch.rand_like(self.local_model.hidden_x)
                n2 = torch.rand_like(self.local_model.hidden_x)
                discriminator_output = self.model.discriminator(self.model.hidden_x)
                local_discriminator_output = self.model.discriminator(self.local_model.hidden_x)
                discriminator_loss = BEcriterion(discriminator_output, zero_label)+BEcriterion(local_discriminator_output, one_label)
                x = F.log_softmax(self.model.hidden_x, dim=-1)
                y = F.softmax(self.local_model.hidden_x, dim=-1)
                loss = criterion(output, target)+BEcriterion(discriminator_output, zero_label)+KLcriterion(x,y)
                a += torch.argmax(output, dim=1).eq(target).sum()
                optimizer.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()

                #本地模型训练
                local_output = self.local_model(data)
                x = F.softmax(self.model.hidden_x+n2, dim=-1)
                y = F.log_softmax(self.local_model.hidden_x+n1, dim=-1)
                discriminator_output = self.local_model.discriminator(self.model.hidden_x)
                local_discriminator_output = self.local_model.discriminator(self.local_model.hidden_x)
                discriminator_loss = BEcriterion(discriminator_output, zero_label) + BEcriterion(local_discriminator_output,one_label)
                local_loss = criterion(local_output, target)+KLcriterion(y, x)+BEcriterion(local_discriminator_output, zero_label)
                b += torch.argmax(local_output, dim=1).eq(target).sum()
                local_optimizer.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                local_loss.backward()
                local_optimizer.step()
        print("local trainings------global accuracy: {:4f}, local accuracy: {:4f}, local_loss:{:4f}, global_model:{:4f}".format( a/total, b/total, local_loss, loss))
        save_file = "../save_local_model_{}/{}.pkl".format(self.args["file_id"],self.client_id)
        save_dict(self.local_model.state_dict(), save_file)

        return SerializationTool.serialize_model(self.local_model)