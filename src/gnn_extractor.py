"""GNN pura-torch (sem PyG/DGL) como extrator SB3.

No (14 = 1 Mario + 5 inimigos + 8 retangulos estaticos) x F=8:
  [dx, dy, w, h, is_mario, is_enemy, is_static, type_norm]
Message passing (2x, grafo completo, media mascarada) + pool global.
Invariante a permutacao dos nos por construcao (o MLP ordenado por
distancia nao e: trocas de slot geram descontinuidades).
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

GLOBAL_DIM = 8
N_NODES = 14
NODE_F = 8
MASK_DIM = N_NODES
OBS_DIM = GLOBAL_DIM + N_NODES * NODE_F + MASK_DIM  # 8+112+14 = 134


class MeanMPNN(nn.Module):
    def __init__(self, h=64, layers=2):
        super().__init__()
        self.enc = nn.Linear(NODE_F, h)
        self.layers = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(),
                                 nn.Linear(h, 256))

    def forward(self, nodes, mask):
        # nodes: (B,N,F), mask: (B,N) 1.0 real / 0.0 pad
        h = torch.relu(self.enc(nodes))
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        for layer in self.layers:
            m = (h * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
            m = m.expand_as(h)
            h = torch.relu(layer(torch.cat([h, m], dim=-1)))
        g = (h * mask.unsqueeze(-1)).sum(dim=1) / denom.squeeze(-1)
        return self.out(torch.cat([g, h[:, 0]], dim=-1))  # pool + no Mario


class GNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        assert observation_space.shape == (OBS_DIM,), observation_space.shape
        self.gnn = MeanMPNN()
        # projecao do vetor global p/ somar (skip connection simples via concat)
        self.head = nn.Sequential(nn.Linear(256 + GLOBAL_DIM, 256), nn.ReLU())

    def forward(self, observations):
        g = observations[:, :GLOBAL_DIM]
        nodes = observations[:, GLOBAL_DIM:GLOBAL_DIM + N_NODES * NODE_F]
        nodes = nodes.view(-1, N_NODES, NODE_F)
        mask = observations[:, GLOBAL_DIM + N_NODES * NODE_F:]
        return self.head(torch.cat([self.gnn(nodes, mask), g], dim=-1))
