import torch.nn as nn
import torch

NET_INPUT_SIZE = 7


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(7, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, state):
        logits = self.layers(state)
        return logits

    def sample_action(self, state):
        logits = self(state)
        c = torch.distributions.Categorical(logits=logits)
        action = int(c.sample().numpy()[0])
        action_prob = float(c.probs[0, action].detach().cpu().numpy())
        return action, action_prob
