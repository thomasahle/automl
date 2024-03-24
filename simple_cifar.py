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
    "aug": {
        "flip": True,
        "translate": 2,
    },
    "net": {
        "widths": {
            "block1": 64,
            "block2": 256,
            "block3": 256,
        },
        "batchnorm_momentum": 0.6,
        "scaling_factor": 1 / 9,
        "tta_level": 2,  # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        # Create an implicit residual via identity initialization
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


def make_net(widths=hyp["net"]["widths"], batchnorm_momentum=hyp["net"]["batchnorm_momentum"]):
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width, widths["block1"], batchnorm_momentum),
        ConvGroup(widths["block1"], widths["block2"], batchnorm_momentum),
        ConvGroup(widths["block2"], widths["block3"], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths["block3"], 10, bias=False),
        Mul(hyp["net"]["scaling_factor"]),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    # net = net.cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net


class KellerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_net()

    def forward(self, x):
        return self.net(x)

    def get_optimizers(self):
        batch_size = hyp["opt"]["batch_size"]
        epochs = hyp["opt"]["train_epochs"]
        momentum = hyp["opt"]["momentum"]
        # Assuming  gradients are constant in time, for Nesterov momentum, the below ratio is how much
        # larger the default steps will be than the underlying per-example gradients. We divide the
        # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
        # of the choice of momentum.
        kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
        lr = hyp["opt"]["lr"] / kilostep_scale  # un-decoupled learning rate for PyTorch SGD
        wd = hyp["opt"]["weight_decay"] * batch_size / kilostep_scale
        lr_biases = lr * hyp["opt"]["bias_scaler"]

        def triangle(steps, start=0, end=0, peak=0.5):
            xp = torch.tensor([0, int(peak * steps), steps])
            fp = torch.tensor([start, 1, end])
            x = torch.arange(1 + steps)
            m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
            b = fp[:-1] - (m * xp[:-1])
            indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
            indices = torch.clamp(indices, 0, len(m) - 1)
            return m[indices] * x + b[indices]

        total_train_steps = 50000 * epochs // batch_size
        lr_schedule = triangle(total_train_steps, start=0.2, end=0.07, peak=0.23)

        model = make_net()

        norm_biases = [p for k, p in model.named_parameters() if "norm" in k and p.requires_grad]
        other_params = [p for k, p in model.named_parameters() if "norm" not in k and p.requires_grad]
        param_configs = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=other_params, lr=lr, weight_decay=wd / lr),
        ]
        optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_schedule[i])
        return optimizer, scheduler, batch_size

    def get_optimizers_(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.85, nesterov=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        batch_size = 512
        return optimizer, scheduler, batch_size


def train(model, train_inputs, train_labels, time_limit):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer, scheduler, batch_size = model.get_optimizers()
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
    return n_items


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
    print(f"{train_inputs.shape=}, {train_labels.shape=}, {test_inputs.shape=}, {test_labels.shape=}")

    return (
        # train_inputs.half().to(device),
        train_inputs.to(device),
        train_labels.to(device),
        # test_inputs.half().to(device),
        test_inputs.to(device),
        test_labels.to(device),
    )


# torch.set_default_dtype(torch.float16)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# device = torch.device("mps")

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
net = KellerNet().to(device)

train_inputs, test_inputs = train_inputs.half(), test_inputs.half()

# train_inputs, test_inputs, net = train_inputs.half(), test_inputs.half(), net.half()
# train_labels, test_labels = train_labels.long(), test_labels.long()
# for layer in net.children():
#     if hasattr(layer, "reset_parameters"):
#         layer.reset_parameters()
#     if hasattr(layer, "dtype"):
#         print(layer.dtype)
#
# print(train_inputs.dtype, train_labels.dtype, test_inputs.dtype, test_labels.dtype)


# print("Compiling model...")
# net = torch.compile(net)
# net = torch.compile(net, mode="max-autotune")
# print("Warmup...")
# for _ in range(3):
#    train(net, train_inputs, train_labels, time_limit=1)
start_time = time.time()
print("Start training...")
n_items = train(net, train_inputs, train_labels, time_limit=5)
print(f"Trained in {time.time() - start_time:.2f} seconds, {n_items / len(train_inputs):.1f} epochs")

# Evaluate on test set
net.eval()
with torch.no_grad():
    outputs = net(test_inputs)
    _, predicted = torch.max(outputs.data, 1)
    total = test_labels.size(0)
    correct = (predicted == test_labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
