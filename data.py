from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
import torch


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

    def __preprocess_data(self, dataset):
        # Create a DataLoader for preprocessing
        preprocess_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=12,
        )

        # Preprocess the data using the DataLoader
        preprocessed_data = []
        for batch in preprocess_loader:
            data, targets = batch
            preprocessed_batch = self.transform(data)
            preprocessed_data.append((preprocessed_batch, targets))

        return preprocessed_data

    def _preprocess_data(self, dataset):
        # Create a DataLoader for preprocessing
        preprocess_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,  # Adjust the number of workers based on your system
            pin_memory=True,
            collate_fn=self._collate_fn,  # Use a custom collate function
        )

        # Preprocess the data using the DataLoader
        preprocessed_data = []
        for batch in preprocess_loader:
            data, targets = batch
            preprocessed_data.append((data, targets))

        return preprocessed_data

    def _collate_fn(self, batch):
        # Custom collate function to handle PIL Image objects
        data, targets = zip(*batch)
        data = [self.transform(img) for img in data]
        data = torch.stack(data)
        targets = torch.tensor(targets)
        return data, targets

    def setup(self, stage=None):
        # Load the preprocessed data into memory
        self.train_data = self.preprocessed_train_data
        self.test_data = self.preprocessed_test_data

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            # num_workers=12,
            # shuffle=True,
            # pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            # num_workers=12,
            batch_size=self.batch_size,
            # pin_memory=True,
        )


class DataModule(LightningDataModule):
    def __init__(self, batch_size, transform, dataset_name, test_run):
        super().__init__()
        self.data_dir = "./"
        self.batch_size = batch_size
        self.transform = transform
        self.dataset_name = dataset_name
        self.test_run = test_run
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
            persistent_workers=not self.test_run,
            num_workers=12 if not self.test_run else 0,
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
