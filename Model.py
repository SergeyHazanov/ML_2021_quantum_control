import torch.nn as nn
import torch

NET_INPUT_SIZE = 8


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(NET_INPUT_SIZE * 2, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 27)  # 9 for a single qubit
        )

    def forward(self, state, previous_state):
        combined = torch.cat([previous_state, state], dim=1)
        logits = self.layers(combined)

        return logits

    def sample_action(self, state, previous_state):
        logits = self(state, previous_state)
        c = torch.distributions.Categorical(logits=logits)
        action = int(c.sample().numpy()[0])
        action_prob = float(c.probs[0, action].detach().cpu().numpy())
        return action, action_prob
