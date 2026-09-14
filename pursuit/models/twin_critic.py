"""Twin LayerNorm MLPs for a shared-reward centralized Q(s, a_joint)."""
from __future__ import annotations

import torch
from torch import nn


class LayerNormQ(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, layers=3):
        super().__init__()
        blocks = []
        last = in_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(last, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            last = hidden_dim
        blocks.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TwinCentralizedQ(nn.Module):
    def __init__(self, state_dim, joint_action_dim, hidden_dim=256, layers=3):
        super().__init__()
        in_dim = int(state_dim) + int(joint_action_dim)
        self.q1 = LayerNormQ(in_dim, hidden_dim, layers)
        self.q2 = LayerNormQ(in_dim, hidden_dim, layers)

    def forward(self, state, joint_action):
        x = torch.cat((state, joint_action), dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, state, joint_action):
        return self.q1(torch.cat((state, joint_action), dim=-1))
