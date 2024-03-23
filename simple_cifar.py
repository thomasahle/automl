import torchvision
import torchvision.transforms as transforms
import time

import torch.optim as optim
import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def make_net():
    act = nn.GELU
    bn = lambda ch: nn.BatchNorm2d(ch)
    conv = lambda ch_in, ch_out: nn.Conv2d(ch_in, ch_out, kernel_size=3, padding="same", bias=False)

    net = nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
        act(),
        nn.Sequential(
            conv(24, 64),
            nn.MaxPool2d(2),
            bn(64),
            act(),
            conv(64, 64),
            bn(64),
            act(),
        ),
        nn.Sequential(
            conv(64, 256),
            nn.MaxPool2d(2),
            bn(256),
            act(),
            conv(256, 256),
            bn(256),
            act(),
        ),
        nn.Sequential(
            conv(256, 256),
            nn.MaxPool2d(2),
            bn(256),
            act(),
            conv(256, 256),
            bn(256),
            act(),
        ),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(256, 10, bias=False),
        Mul(1 / 9),
    )
    net[0].weight.requires_grad = False
    return net


def train(device, train_inputs, train_labels, time_limit):
    # model = Net().to(device)
    model = make_net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    batch_size = 256
    n_items = 0
    start_time = time.time()
    while time.time() - start_time < time_limit:
        perm = torch.randperm(len(train_inputs))
        train_inputs = train_inputs[perm]
        train_labels = train_labels[perm]
        for i in range(0, len(train_inputs), batch_size):
            optimizer.zero_grad()
            outputs = model(train_inputs[i : i + batch_size])
            loss = criterion(outputs, train_labels[i : i + batch_size])
            loss.backward()
            optimizer.step()
            n_items += batch_size
            if time.time() - start_time >= time_limit:
                break
        scheduler.step()
    return model, n_items


def make_data(device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    traindata = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
    testdata = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
    train_inputs, train_labels = next(iter(traindata))
    test_inputs, test_labels = next(iter(testdata))
    return train_inputs.to(device), train_labels.to(device), test_inputs.to(device), test_labels.to(device)


# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

# Make the data
print("Loading data")
start_time = time.time()
train_inputs, train_labels, test_inputs, test_labels = make_data(device)
print(
    f"Loaded {len(train_inputs)} training and {len(test_inputs)} test examples in {time.time() - start_time:.2f} seconds"
)

# Train the model
start_time = time.time()
print("Start training...")
model, n_items = train(device, train_inputs, train_labels, time_limit=5)
print(f"Trained in {time.time() - start_time:.2f} seconds, {n_items / len(train_inputs):.1f} epochs")

# Evaluate on test set
model.eval()
with torch.no_grad():
    outputs = model(test_inputs)
    _, predicted = torch.max(outputs.data, 1)
    total = test_labels.size(0)
    correct = (predicted == test_labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
