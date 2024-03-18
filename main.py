from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import dspy
import pydantic, textwrap, re
import torch.nn.functional as F
import torch
import random
import numpy as np
import argparse
import dotenv, os
from datetime import timedelta

import default_progs


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["mnist", "cifar"])
parser.add_argument("train_time", type=int, default=3)
parser.add_argument("--devices", type=str)
args = parser.parse_args()


torch.set_float32_matmul_precision("medium")


class DataModule(LightningDataModule):
    def __init__(self, batch_size, transform, dataset_name):
        super().__init__()
        self.data_dir = "./"
        self.batch_size = batch_size
        self.transform = transform
        self.dataset_name = dataset_name

    def prepare_data(self):
        if self.dataset_name == "mnist":
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)
        elif self.dataset_name == "cifar":
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if self.dataset_name == "mnist":
            self.train_dataset = MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.test_dataset = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        elif self.dataset_name == "cifar":
            self.train_dataset = CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
            self.test_dataset = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=12,
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=12,
            pin_memory=True,
        )


# We define the test-step outside of the MNISTModel, so the LM can't cheat
def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    preds = y_hat.argmax(dim=1)
    acc = (preds == y).float().mean()
    metrics = {"test_loss": loss, "test_acc": acc}
    self.log_dict(metrics)
    return metrics


class BatchCounterCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.batch_count = 0

    def on_train_batch_end(self, *args):
        self.batch_count += 1


def compute_accuracy(model_class, dataset, train_time_seconds: int, test_run=False):
    # Instantiate the model and data module
    model = model_class()
    model.test_step = lambda *args: test_step(model, *args)
    batch_size = getattr(model, "batch_size", 64)
    transform = getattr(model, "transform", transforms.Compose([transforms.ToTensor()]))
    data_module = DataModule(
        batch_size=batch_size, transform=transform, dataset_name=dataset
    )

    if test_run:
        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            devices=1,
        )
        trainer.fit(model, datamodule=data_module)
        return None

    try:
        batch_counter = BatchCounterCallback()
        trainer = pl.Trainer(
            max_time=timedelta(seconds=train_time_seconds),
            accelerator="gpu",
            devices=args.devices,
            callbacks=[batch_counter],  # Register the callback
        )
        trainer.fit(model, datamodule=data_module)
        res = trainer.test(
            model, dataloaders=data_module.test_dataloader(), verbose=False
        )
    except Exception as e:
        print(f"Warning: Got exception {e}")
        return 0, 0, 0

    n_examples = batch_counter.batch_count * batch_size
    fractional_epochs = batch_counter.batch_count / len(data_module.train_dataloader())
    return res[0]["test_acc"], n_examples, fractional_epochs


def strip_ticks(s):
    s = s.strip()
    if s.startswith("```python\n"):
        assert s.endswith("```")
        return s[10:-3]
    return s


def run_code_and_get_class(code):
    namespace = {}
    exec(code, namespace)
    return namespace["ImageModel"]


def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class MySignature(dspy.Signature):
    __doc__ = textwrap.dedent(
        f"""
    Write a new python lightning class to get the best score on {args.dataset} given {args.train_time} seconds training time. I will give you some examples, and you should use your creativity to suggest a better model, that will achieve higher accuracy than any of the previous models.

    You can try things such as:
    - Changing the batch size or learning rate
    - Trying diferent architectures
    - Using different activation functions
    - Using different optimizers
    - Using different data augmentation transformations
    - Using different normalization techniques
    - Using different loss functions
    - Using different learning rate schedules
    - Using different weight initialization
    - Using different regularization techniques
    - Using different dropout rates
    - Using different layer types and sizes
    - Using different hyper parameters
    - Making the model run faster

    Also take care to:
    - Not use too much memory
    - Make every new program different from the previous ones
    - Don't import pretrained models
    """
    )

    score: float = dspy.InputField(desc="The accuracy the model should get")
    analysis: str = dspy.OutputField(
        desc="Short analysis of what the previous models did that worked well or didn't work well. How can you improve on them? Answer in plain text, no Markdown"
    )
    program: str = dspy.OutputField(
        desc="A python lightning class, called ImageModel, you can include imports, but no other code."
    )
    explanation: str = dspy.OutputField(desc="Short explanation of what's different")
    # number_of_parameters: int = dspy.OutputField(desc="Total number of parameters in model")
    # number_of_exampled_processed: int = dspy.OutputField(desc=f"Number of examples processed in {args.train_time} seconds")

    @pydantic.field_validator("analysis")
    def check_for_score(cls, s):
        # Sometimes the model invents its own score. Remove that.
        return re.sub("Score: [\d\.]+", "", s)

    @pydantic.field_validator("program")
    def check_syntax(cls, v):
        v = strip_ticks(v)
        lines = v.split("\n")
        allowed = ["import", "from", "class", " ", "#"]
        for line in lines:
            if line and not any(line.startswith(prefix) for prefix in allowed):
                raise ValueError(
                    f"Don't write any code besides the class. You wrote {repr(line)}"
                )
        if "torchvision.models" in v:
            raise ValueError(f"Don't import torchvision.models")
        if "class ImageModel" not in v:
            raise ValueError(f"You must define one class named ImageModel")
        try:
            # Attempt to compile the code snippet
            compile(v, "<string>", "exec")
            # Attempt to run the code snippet
            Model = run_code_and_get_class(v)
            total_params, _ = get_model_parameters(Model())
            if total_params > 10 ** 7:
                raise ValueError(
                    f"You used {total_params} parameters. Please keep it under 1,000,000"
                )
            _ = compute_accuracy(Model, args.dataset, 0, test_run=True)
        except Exception as e:
            raise ValueError(f"Code did not run: {e}")
        return v


def model_producer(dataset, train_time):
    solver = dspy.TypedPredictor(MySignature, explain_errors=True, max_retries=10)
    data_module = DataModule(
        batch_size=64,
        transform=transforms.Compose([transforms.ToTensor()]),
        dataset_name=dataset,
    )
    used_demo_subsets = set()

    while True:
        # Sample demos to use
        examples = solver.predictor.demos[:]
        ps = [ex.score ** 2 + 0.01 for ex in examples]
        js = np.random.choice(
            range(len(examples)),
            min(30, len(examples)),
            replace=False,
            p=[p / sum(ps) for p in ps],
        )
        demo_subset = tuple(sorted(js))

        # Skip if the demo subset has already been used
        if demo_subset in used_demo_subsets:
            continue
        used_demo_subsets.add(demo_subset)

        # Create a new TypedPredictor with the sampled demos
        solver_with_demos = dspy.TypedPredictor(
            MySignature, explain_errors=True, max_retries=10
        )
        solver_with_demos.predictor.demos = [examples[j] for j in js]

        pred = solver_with_demos(score=0.99)
        Model = run_code_and_get_class(strip_ticks(pred.program))
        batch_size = getattr(Model, "batch_size", None)
        transform = getattr(Model, "transform", None)
        score, n_examples, n_epochs = compute_accuracy(
            Model, data_module, train_time, batch_size=batch_size, transform=transform
        )
        result_queue.put((pred, score, n_examples, n_epochs, Model))

        # Update the solver's demos based on the scored examples
        while not score_queue.empty():
            example = score_queue.get()
            solver.predictor.demos.append(example)
            solver.predictor.demos.sort(key=lambda e: e.score)
            solver.predictor.demos = solver.predictor.demos[
                -100:
            ]  # Keep only the top 100 examples


def main():
    random.seed(42)
    torch.manual_seed(42)
    dotenv.load_dotenv(os.path.expanduser("~/.env"))
    lm = dspy.OpenAI(model="gpt-4-turbo-preview", max_tokens=4000)
    # lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    dspy.settings.configure(lm=lm)

    if args.dataset == "mnist":
        program0 = default_progs.mnist
    elif args.dataset == "cifar":
        program0 = default_progs.cifar

    solver = dspy.TypedPredictor(MySignature, explain_errors=True, max_retries=2)
    examples = []
    actual_scores = []
    for i in range(100):
        print("Suggesting new model...")
        if i > 0:
            try:
                pred = solver(score=0.99)
            except ValueError:
                lm.inspect_history(n=1)
                raise
        else:
            pred = dspy.Example(
                program=program0,
                analysis="Baseline model",
                explanation=f"",
            )
        print(f"Anlysis: {pred.analysis}")
        print("Program:", pred.program)
        print("Explanation:", pred.explanation)
        Model = run_code_and_get_class(strip_ticks(pred.program))
        score, n_examples, n_epochs = compute_accuracy(
            Model, args.dataset, args.train_time
        )
        print(f"Actual score: {score:.2f}")
        print(f"Examples processed: {n_examples}")
        model = Model()
        total_params, _ = get_model_parameters(model)
        examples.append(
            dspy.Example(
                program=pred.program,
                analysis=pred.analysis[:100] + "...",
                score=score,
                explanation=f"Model with {total_params} parameters. Speed: {n_examples/args.train_time:.2f} examples per second. Completed {n_epochs:.2f} epochs.",
            )
        )
        # Limit to 30 examples
        ps = [ex.score ** 2 + 0.01 for ex in examples]
        js = np.random.choice(
            range(len(examples)),
            min(30, len(examples)),
            replace=False,
            p=[p / sum(ps) for p in ps],
        )
        demos = [examples[j] for j in js]
        # Sort to have monotonically increasing examples
        # demos = examples[:]
        demos.sort(key=lambda e: e.score)
        solver.predictor.demos = demos
        actual_scores.append(score)
        print(actual_scores)

    lm.inspect_history(n=2)

    print("Best Program:")
    for demo in solver.predictor.demos:
        if demo.score == max(actual_scores):
            print(demo.program)
            break

    import matplotlib.pyplot as plt

    plt.plot(actual_scores)
    plt.show()


if __name__ == "__main__":
    main()
