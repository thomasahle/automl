import argparse
import math
import torchvision
import torchvision.transforms as transforms
import time
import torch

sample_nets = [
    """
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 16x16x16
        x = self.pool(F.relu(self.conv2(x))) # 32x8x8
        x = x.reshape(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_optimizers(self):
        batch_size = 256
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        return optimizer, scheduler, loss_fn, batch_size
""",
    # Based on Keller's Net 94
    """
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True), # 24x31x31
            nn.GELU(),
            self._make_conv_group(24, 64),   # 64x15x15
            self._make_conv_group(64, 256),  # 256x7x7
            self._make_conv_group(256, 256), # 256x3x3
            nn.MaxPool2d(3),                 # 256x1x1
            nn.Flatten(),
            nn.Linear(256, 10, bias=False),
        )

    def _make_conv_group(self, channels_in, channels_out):
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
        optimizer = optim.AdamW(self.parameters(), lr=0.01, fused=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000 * 10 / batch_size)
        loss_fn = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.2)
        return optimizer, scheduler, loss_fn, batch_size
""",
    # Based on Keller's Net 96
    """
import torch
import torch.nn as nn
import torch.optim as optim
from math import ceil

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding="same", bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding="same", bias=False)
        self.norm2 = nn.BatchNorm2d(channels_out)
        self.conv3 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding="same", bias=False)
        self.norm3 = nn.BatchNorm2d(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activ(x)
        x += x0
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
            nn.GELU(),
            ConvGroup(24, 64),
            ConvGroup(64, 256),
            ConvGroup(256, 256),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(256, 10, bias=False),
        )

    def forward(self, x):
        return self.net(x) / 9

    def get_optimizers(self):
        epochs = 10
        total_train_steps = ceil(50000 * epochs)
        batch_size = 1024
        momentum = 0.85
        kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
        lr = 9 / kilostep_scale
        wd = 0.0153 * batch_size / kilostep_scale
        lr_biases = lr * 64

        norm_biases = [p for name, p in self.named_parameters() if 'norm' in name and p.requires_grad]
        other_params = [p for name, p in self.named_parameters() if 'norm' not in name and p.requires_grad]
        
        optimizer = optim.SGD([
            {'params': norm_biases, 'lr': lr_biases, 'weight_decay': wd / lr_biases},
            {'params': other_params, 'lr': lr, 'weight_decay': wd}
        ], momentum=momentum, nesterov=True)

        lr_schedule = self.triangle_lr_schedule(total_train_steps, start=0.2, end=0.07, peak=0.23)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
        
        return optimizer, scheduler, loss_fn, batch_size

    def triangle_lr_schedule(self, steps, start=0, end=0, peak=0.5):
        def schedule(current_step):
            xp = [0, int(peak * steps), steps]
            fp = [start, 1, end]
            if current_step < xp[1]:
                return fp[0] + (fp[1] - fp[0]) * (current_step - xp[0]) / (xp[1] - xp[0])
            else:
                return fp[1] + (fp[2] - fp[1]) * (current_step - xp[1]) / (xp[2] - xp[1])
        return schedule
""",
]


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

        model.eval()
        with torch.no_grad():
            # During evaluation, we use left-right flipping TTA, in which the network is run on both a given test image
            # and its mirror, and inferences are made based on the average of the two outputs.
            corrects = 0
            for e in range(2):
                if e == 1:
                    test_inputs = torch.flip(test_inputs, [3])
                for i in range(0, len(train_inputs), batch_size):
                    outputs = model(test_inputs[i : i + batch_size])
                    _, predicted = torch.max(outputs.data, 1)
                    corrects += (predicted == test_labels[i : i + batch_size]).sum().item()
            accuracy = corrects / (2 * test_labels.size(0))
            results.append([total_items / len(train_labels), train_loss, accuracy, total_time_seconds])
            print(f"{results[-1][0]:10.2f}{results[-1][1]:13.4f}{results[-1][2]*100:12.2f}%{results[-1][3]:9.2f}s")

    return results


def make_data(dataset, test_run=False):
    if test_run:
        if dataset == "mnist":
            inputs = torch.randn(10000, 1, 28, 28)
            labels = torch.randint(0, 10, (10000,))
        elif dataset == "cifar10":
            inputs = torch.randn(10000, 3, 32, 32)
            labels = torch.randint(0, 10, (10000,))
        return inputs[:9000], labels[:9000], inputs[9000:], labels[9000:]

    # We can do some data augmentation here.
    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    if dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}")

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

    # TODO: Do whitening here?
    def batch_normalize_images(input_images):
        return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    train_inputs = batch_normalize_images(train_inputs)
    test_inputs = batch_normalize_images(test_inputs)

    return (train_inputs, train_labels, test_inputs, test_labels)


# This is totally unsafe. Run at your own risk.
def run_code_and_get_class(code, class_name):
    namespace = {}
    exec(code, namespace)
    return namespace[class_name]


def main(
    program,
    device,
    dataset,
    time_limit=5,
    test_run=False,
    compile=False,
    n_runs=3,
):
    model_class = run_code_and_get_class(program, "Net")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("Loading data")
    start_time = time.time()
    train_inputs, train_labels, test_inputs, test_labels = make_data(dataset, test_run=test_run)
    train_labels, test_labels = train_labels.to(device), test_labels.to(device)
    train_inputs = train_inputs.to(dtype).to(memory_format=torch.channels_last).to(device)
    test_inputs = test_inputs.to(dtype).to(memory_format=torch.channels_last).to(device)
    print(
        f"Loaded {len(train_inputs)} training and {len(test_inputs)} "
        f"test examples in {time.time() - start_time:.2f} seconds"
    )

    if compile:
        print("Compiling model...")
        net = model_class().to(dtype).to(device).to(memory_format=torch.channels_last)
        net = torch.compile(net)
        net = torch.compile(net, mode="max-autotune")
        print("Warmup...")
        for _ in range(3):
            train(net, train_inputs, train_labels, time_limit=1)

    print("Warmup...")
    net = model_class().to(dtype).to(device).to(memory_format=torch.channels_last)
    train(net, train_inputs, train_labels, test_inputs, test_labels, time_limit=1)

    if test_run:
        return 0, 0

    accuracies = []
    train_losses = []
    for i in range(n_runs):
        print(f"\nRun {i+1}:")
        net = model_class().to(dtype).to(device).to(memory_format=torch.channels_last)
        results = train(net, train_inputs, train_labels, test_inputs, test_labels, time_limit=time_limit)
        accuracies.append([result[2] for result in results])
        train_losses.append([result[1] for result in results])

    # Pad the accuracies and training losses to have same length
    length = max(len(acc) for acc in accuracies)
    accuracies = [acc + [acc[-1]] * (length - len(acc)) for acc in accuracies]
    train_losses = [tl + [tl[-1]] * (length - len(tl)) for tl in train_losses]

    # Compute the standard deviation of the accuracy and training loss
    accuracies = torch.tensor(accuracies)
    train_losses = torch.tensor(train_losses)
    mean_accuracy = torch.mean(accuracies, axis=0)
    std_accuracy = torch.std(accuracies, axis=0)
    mean_train_loss = torch.mean(train_losses, axis=0)
    std_train_loss = torch.std(train_losses, axis=0)

    print("\nTraining Statistics:")
    print(f"{'Epoch':>5}{'Train Loss':>25}{'Accuracy (%)':>25}")
    print(f"{'-'*5:>5}{'-'*25:>25}{'-'*25:>25}")
    for i in range(len(mean_accuracy)):
        print(
            f"{i+1:5}"
            f"{mean_train_loss[i]:13.4f} +/- {std_train_loss[i]:.4f}"
            f"{mean_accuracy[i]*100:12.2f} +/- {std_accuracy[i]*100:5.2f}%"
        )

    acc = mean_accuracy[-1].item()
    std = std_accuracy[-1].item()
    return acc, 0 if math.isnan(std) else std


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    parser = argparse.ArgumentParser()
    parser.add_argument("program", type=int, default=0)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--time-limit", type=int, default=5)
    args = parser.parse_args()

    main(sample_nets[args.program], device, "cifar10", time_limit=args.time_limit, test_run=args.test_run)
