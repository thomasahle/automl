imports = """
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
"""


suffix = """
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
"""

template = """
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms
...

class ModelClassName(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Define layers here
        ...

        # batch_size and transform are parameters used by the data loader
        self.batch_size = 64
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ...
        ])

    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        ...
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
"""

mnist = f"""
{imports}
class ImageModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # MNIST images are (1, 28, 28) (channels, width, height)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)
        # These parameters are used by the data loader
        self.batch_size = 64
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x
{suffix}
"""

cifar = f"""
{imports}
class ImageModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
        # These parameters are used by the data loader
        self.batch_size = 64
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
{suffix}
"""
