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


class KellerNet(nn.Module):
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    class Mul(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return x * self.scale

    def __init__(self):
        super().__init__()
        act = nn.GELU
        bn = lambda ch: nn.BatchNorm2d(ch)
        conv = lambda ch_in, ch_out: nn.Conv2d(ch_in, ch_out, kernel_size=3, padding="same", bias=False)

        def make_layer(ch_in, ch_out):
            return nn.Sequential(
                conv(ch_in, ch_out),
                bn(ch_out),
                act(),
                conv(ch_out, ch_out),
                bn(ch_out),
                act(),
            )

        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
            act(),
            make_layer(24, 64),
            make_layer(64, 256),
            make_layer(256, 256),
            nn.MaxPool2d(3),
            self.Flatten(),
            nn.Linear(256, 10, bias=False),
            self.Mul(1 / 9),
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        batch_size = 500
        hyp = {
            "opt": {
                "train_epochs": 9.9,
                "batch_size": 1024,
                "lr": 11.5,  # learning rate per 1024 examples
                "momentum": 0.85,
                "weight_decay": 0.0153,  # weight decay per 1024 examples (decoupled from learning rate)
                "bias_scaler": 64.0,  # scales up learning rate (but not weight decay) for BatchNorm biases
                "label_smoothing": 0.2,
                "ema": {
                    "start_epochs": 3,
                    "decay_base": 0.95,
                    "decay_pow": 3.0,
                    "every_n_steps": 5,
                },
                "whiten_bias_epochs": 3,  # how many epochs to train the whitening layer bias before freezing
            },
        }

        def triangle(steps, start=0, end=0, peak=0.5):
            xp = torch.tensor([0, int(peak * steps), steps])
            fp = torch.tensor([start, 1, end])
            x = torch.arange(1 + steps)
            m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
            b = fp[:-1] - (m * xp[:-1])
            indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
            indices = torch.clamp(indices, 0, len(m) - 1)
            return m[indices] * x + b[indices]

        total_train_steps = 50_000 * 5
        lr_schedule = triangle(total_train_steps, start=0.2, end=0.07, peak=0.23)
        momentum = hyp["opt"]["momentum"]
        kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
        lr = hyp["opt"]["lr"] / kilostep_scale  # un-decoupled learning rate for PyTorch SGD
        wd = hyp["opt"]["weight_decay"] * batch_size / kilostep_scale
        lr_biases = lr * hyp["opt"]["bias_scaler"]

        norm_biases = [p for k, p in model.named_parameters() if "norm" in k and p.requires_grad]
        other_params = [p for k, p in model.named_parameters() if "norm" not in k and p.requires_grad]
        param_configs = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=other_params, lr=lr, weight_decay=wd / lr),
        ]
        optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_schedule[i])
        return optimizer, scheduler, batch_size


def train(device, train_inputs, train_labels, time_limit):
    # model = Net().to(device)
    model = KellerNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler, batch_size = model.configure_optimizers()
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
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    traindata = torch.utils.data.DataLoader(trainset, batch_size=2 * len(trainset), shuffle=False, num_workers=0)
    testdata = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
    train_inputs, train_labels = next(iter(traindata))
    test_inputs, test_labels = next(iter(testdata))

    std, mean = torch.std_mean(train_inputs, dim=(0, 2, 3))

    def batch_normalize_images(input_images):
        return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    train_inputs = batch_normalize_images(train_inputs)
    test_inputs = batch_normalize_images(test_inputs)

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
