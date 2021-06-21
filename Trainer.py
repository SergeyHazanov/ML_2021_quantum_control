from torch.utils.data import DataLoader
from DataLoader import CustomLoader
import torch


class Trainer:
    """
    Supervises the the training of a given network.
    """

    def __init__(self, net):
        self.network = net
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    def save_net(self, path):
        torch.save(self.network.state_dict(), path)

    def train(self, n_epochs, reps):
        for epoch in range(n_epochs):
            # This is inside the loop because we restart the information each iteration
            data_loader = DataLoader(CustomLoader(reps))
            self.network.train()  # put the net into "training mode"
            for x, y in data_loader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # add the basic training loop here
                self.optimizer.zero_grad()
                output = self.network(x)

                # TODO: calculate the actualy fidelity between the target
                #       state y
                fidelity = 3000000000
                loss = self.loss_func(output, y)
                loss.backward()


                self.optimizer.step()
            self.network.eval()  # put the net into evaluation mode

            # train_acc, train_loss = compute_accuracy_and_loss(training_dataloader, net)
            # valid_acc, valid_loss = compute_accuracy_and_loss(valid_dataloader, net)

            # training_loss_vs_epoch.append(train_loss)
            # training_acc_vs_epoch.append(train_acc)
            #
            # validation_acc_vs_epoch.append(valid_acc)
            #
            # validation_loss_vs_epoch.append(valid_loss)

            # save the model if the validation loss has decreased
            # if len(validation_loss_vs_epoch) == 1 or validation_loss_vs_epoch[-2] > validation_loss_vs_epoch[-1]:
            #     torch.save(net.state_dict(), 'trained_model.pt')