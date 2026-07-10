import torch.nn as nn
import torch.nn.functional as F


class DuelCNNImproved(nn.Module):
    """
    Improved CNN with Duel Algo. https://arxiv.org/abs/1511.06581
    Updated to match the description with 512 hidden units and LayerNorm for stability
    """

    def __init__(self, h, w, output_size):
        super(DuelCNNImproved, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Action layer (Advantage stream)
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=512)
        self.Alayer_norm1 = nn.LayerNorm(512)  # LayerNorm instead of BatchNorm1d
        self.Alinear2 = nn.Linear(in_features=512, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=512)
        self.Vlayer_norm1 = nn.LayerNorm(512)  # LayerNorm instead of BatchNorm1d
        self.Vlinear2 = nn.Linear(in_features=512, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        # Advantage stream
        Ax = F.relu(self.Alayer_norm1(self.Alinear1(x)))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        # Value stream
        Vx = F.relu(self.Vlayer_norm1(self.Vlinear1(x)))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        # Combine value and advantage
        # Mean over actions per sample (dueling aggregation, Wang et al. 2016).
        # A global Ax.mean() would couple samples across the batch.
        q = Vx + (Ax - Ax.mean(dim=1, keepdim=True))

        return q
