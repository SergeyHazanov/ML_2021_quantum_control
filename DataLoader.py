from torch.utils.data import Dataset
import torch

TARGET_STATE = 1


class CustomLoader(Dataset):
    def __init__(self, length):
        # a vector of Bernoulli ~(1/2) distributions representing
        # the initial states. Either |0> or |1>.
        self.length = length
        self.X = torch.randint(0, 2, (length,))

        # For simplicity we will train only on |0> for now.
        self.X = self.X * 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.X[idx]
        y = TARGET_STATE
        return x, y
