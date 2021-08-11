import torch.nn as nn
import torch

NET_INPUT_SIZE = 7


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(NET_INPUT_SIZE * 2, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, state, prev_state):
        combined = torch.cat([prev_state, state], dim=1)
        logits = self.layers(combined)
        return logits

    def sample_action(self, state, prev_state):
        logits = self(state, prev_state)
        c = torch.distributions.Categorical(logits=logits)
        action = int(c.sample().numpy()[0])
        action_prob = float(c.probs[0, action].detach().cpu().numpy())
        return action, action_prob
