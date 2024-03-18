from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10


class MemoryDataModule(LightningDataModule):
    def __init__(self, batch_size, transform, dataset_name):
        super().__init__()
        self.data_dir = "./"
        self.batch_size = batch_size
        self.transform = transform
        self.dataset_name = dataset_name
        if dataset_name == "mnist":
            self.dataset_class = MNIST
        elif dataset_name == "cifar10":
            self.dataset_class = CIFAR10
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def prepare_data(self):
        # Download the data
        train_dataset = self.dataset_class(self.data_dir, train=True, download=True)
        test_dataset = self.dataset_class(self.data_dir, train=False, download=True)

        # Preprocess the data
        self.preprocessed_train_data = self._preprocess_data(train_dataset)
        self.preprocessed_test_data = self._preprocess_data(test_dataset)

    def _preprocess_data(self, dataset):
        # Apply your preprocessing steps here
        preprocessed_data = []
        for data, target in dataset:
            preprocessed_data.append((self.transform(data), target))
        return preprocessed_data

    def setup(self, stage=None):
        # Load the preprocessed data into memory
        self.train_data = self.preprocessed_train_data
        self.test_data = self.preprocessed_test_data

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            pin_memory=True,
        )


class DataModule(LightningDataModule):
    def __init__(self, batch_size, transform, dataset_name):
        super().__init__()
        self.data_dir = "./"
        self.batch_size = batch_size
        self.transform = transform
        self.dataset_name = dataset_name
        if dataset_name == "mnist":
            self.dataset_class = MNIST
        elif dataset_name == "cifar10":
            self.dataset_class = CIFAR10
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = self.dataset_class(self.data_dir, train=True, transform=self.transform)
        self.test_dataset = self.dataset_class(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=3,
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            persistent_workers=False,
            num_workers=3,
            pin_memory=True,
        )
