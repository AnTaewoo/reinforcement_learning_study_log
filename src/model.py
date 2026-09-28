import torch.nn as nn
import torch.nn.functional as f


class CDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(CDQN, self).__init__()
        self.conv1 = nn.Conv2d(state_size, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.flat = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(6400, 512)
        self.fout = nn.Linear(512, action_size)

    def forward(self, state):
        x = f.max_pool2d(f.relu(self.conv1(state)), 2)
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = f.max_pool2d(f.relu(self.conv3(x)), 2)
        x = self.fc1(self.flat(x))
        return self.fout(f.relu(x))
