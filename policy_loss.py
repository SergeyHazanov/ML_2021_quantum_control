import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import qutip as qt


class PolicyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps_clip = 0.1

    def forward(self, logits, action, action_prob, discounted_rewards):
        new_action_prob = F.softmax(logits, dim=1).gather(1, action.unsqueeze(1)).view(-1)
        ratio = new_action_prob / action_prob

        loss1 = ratio * discounted_rewards
        loss2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * discounted_rewards
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss

    def state2vec(self, qubit=1):
        """
        Bloch vector representation of the state self.state
        :return: list of expectation values for the state self.state.
        """
        x, y, z = qt.expect(qt.sigmax(), qt.ptrace(self.state, qubit)),\
                  qt.expect(qt.sigmay(), qt.ptrace(self.state, qubit)),\
                  qt.expect(qt.sigmaz(), qt.ptrace(self.state, qubit))
        return [x, y, z]

    def fidelity(self, qubit=1):
        """
        fidelity with respect to |1> (which is the target state)
        :return: F(|ψ>,|1>) of type float.
        """
        return qt.fidelity(qt.ptrace(self.state, qubit), qt.basis(2, 1))