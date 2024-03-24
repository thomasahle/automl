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

    def get_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        batch_size = 256
        return optimizer, scheduler, batch_size


## KellerNet


class KellerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
            nn.GELU(),
            self.make_conv_group(24, 64),
            self.make_conv_group(64, 256),
            self.make_conv_group(256, 256),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(256, 10, bias=False),
        )

    def make_conv_group(self, channels_in, channels_out):
        return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding="same", bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channels_out),
            nn.GELU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(channels_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x) / 9

    def get_optimizers(self):
        batch_size = 1024
        optimizer = optim.Adam(self.parameters(), lr=0.01, fused=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000 * 10 / batch_size)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        return optimizer, scheduler, loss_fn, batch_size


def train(model, train_inputs, train_labels, test_inputs, test_labels, time_limit):
    optimizer, scheduler, criterion, batch_size = model.get_optimizers()

    print(f"{'Epoch':>10}{'Train Loss':>13}{'Test Acc':>13}{'Time':>10}")
    print(f"{'-'*10}{'-'*13}{'-'*13}{'-'*10}")

    total_time_seconds = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    results = []
    total_items = 0
    while total_time_seconds < time_limit:
        perm = torch.randperm(len(train_inputs))
        train_inputs = train_inputs[perm]
        train_labels = train_labels[perm]

        starter.record()
        model.train()

        approx_time = time.time()
        train_loss = 0.0
        train_items = 0
        for i in range(0, len(train_inputs), batch_size):
            optimizer.zero_grad()
            outputs = model(train_inputs[i : i + batch_size])
            loss = criterion(outputs, train_labels[i : i + batch_size])
            loss.backward()
            optimizer.step()

            this_batch_size = len(train_inputs[i : i + batch_size])
            total_items += this_batch_size
            train_items += this_batch_size
            train_loss += loss.item()
            if total_time_seconds + time.time() - approx_time >= time_limit:
                break
        train_loss /= train_items
        scheduler.step()

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        net.eval()
        with torch.no_grad():
            outputs = net(test_inputs)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
            results.append([total_items / len(train_labels), train_loss, accuracy, total_time_seconds])
            print(f"{results[-1][0]:10.2f}{results[-1][1]:13.4f}{results[-1][2]*100:12.2f}%{results[-1][3]:9.2f}s")

    return results


def make_data(device):
    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    traindata = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
    testdata = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
    test_inputs, test_labels = next(iter(testdata))
    train_inputs, train_labels = [], []
    for _ in range(1):
        ti, tl = next(iter(traindata))
        train_inputs.append(ti)
        train_labels.append(tl)
    train_inputs = torch.cat(train_inputs)
    train_labels = torch.cat(train_labels)

    std, mean = torch.std_mean(train_inputs, dim=(0, 2, 3))

    def batch_normalize_images(input_images):
        return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    train_inputs = batch_normalize_images(train_inputs)
    test_inputs = batch_normalize_images(test_inputs)

    return (
        train_inputs.to(torch.bfloat16).to(device),
        train_labels.to(device),
        test_inputs.to(torch.bfloat16).to(device),
        test_labels.to(device),
    )


# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Make the data
print("Loading data")
start_time = time.time()
train_inputs, train_labels, test_inputs, test_labels = make_data(device)
print(
    f"Loaded {len(train_inputs)} training and {len(test_inputs)} "
    f"test examples in {time.time() - start_time:.2f} seconds"
)


# net = Net().to(device)
print("Creating model...")

# print("Compiling model...")
# net = torch.compile(net)
# net = torch.compile(net, mode="max-autotune")
# print("Warmup...")
# for _ in range(3):
#    train(net, train_inputs, train_labels, time_limit=1)

print("Warmup...")
net = KellerNet().to(torch.bfloat16).to(device).to(memory_format=torch.channels_last)
train(net, train_inputs, train_labels, test_inputs, test_labels, time_limit=1)

accuracies = []
for i in range(3):
    print(f"\nRun {i+1}:")
    net = KellerNet().to(torch.bfloat16).to(device).to(memory_format=torch.channels_last)
    results = train(net, train_inputs, train_labels, test_inputs, test_labels, time_limit=5)
    accuracies.append([result[2] for result in results])

# Compute the standard deviation of the accuracy
accuracies = torch.tensor(accuracies)
mean_accuracy = torch.mean(accuracies, axis=0)
std_accuracy = torch.std(accuracies, axis=0)

print("\nAccuracy Statistics:")
print(f"{'Epoch':>10}{'Mean Accuracy (%)':>20}{'Std Deviation':>20}")
print(f"{'-'*10:>10}{'-'*20:>20}{'-'*20:>20}")
for i in range(len(mean_accuracy)):
    print(f"{i+1:10}{mean_accuracy[i]:20.2f}{std_accuracy[i]:20.2f}")
