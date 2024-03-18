import pytorch_lightning as pl
from torchvision import transforms
from datetime import timedelta
import torch.nn.functional as F

from data import DataModule


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


# For counting the number of batches we train on
class BatchCounterCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.batch_count = 0

    def on_train_batch_end(self, *args):
        self.batch_count += 1


# This is totally unsafe. Run at your own risk.
def run_code_and_get_class(code):
    namespace = {}
    exec(code, namespace)
    return namespace["ImageModel"]


def strip_ticks(s):
    s = s.strip()
    if s.startswith("```python\n"):
        assert s.endswith("```")
        return s[10:-3]
    return s


def compute_accuracy(model_class: str, args, test_run=False):
    model = model_class()
    batch_size = getattr(model, "batch_size", 64)
    transform = getattr(model, "transform", transforms.Compose([transforms.ToTensor()]))
    data_module = DataModule(batch_size=batch_size, transform=transform, dataset_name=args.dataset)

    try:
        if test_run:
            trainer = pl.Trainer(
                fast_dev_run=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                accelerator="cpu",
                devices=1,
            )
            # We don't wrap this in a try-except block, because we want to see the error
            trainer.fit(model, datamodule=data_module)
            return None

        try:
            batch_counter = BatchCounterCallback()
            trainer = pl.Trainer(
                max_time=timedelta(seconds=args.train_time),
                accelerator=args.accelerator,
                devices=1 if args.accelerator == "cpu" else args.devices,
                callbacks=[batch_counter],  # Register the callback
                enable_checkpointing=False,
            )
            trainer.fit(model, datamodule=data_module)
            model.test_step = lambda *args: test_step(model, *args)  # Attach test_step
            res = trainer.test(model, dataloaders=data_module.test_dataloader(), verbose=False)
        except Exception as e:
            print(f"Warning: Got exception {e}")
            return 0, 0, 0

        n_examples = batch_counter.batch_count * batch_size
        fractional_epochs = batch_counter.batch_count / len(data_module.train_dataloader())
        return res[0]["test_acc"], n_examples, fractional_epochs
    finally:
        data_module.teardown("fit")


def model_tester(args, task_queue, result_queue, worker_idx):
    while True:
        pidx, program = task_queue.get()
        try:
            qsize = task_queue.qsize()
        except NotImplementedError:
            qsize = -1  # Not implemented on MacOS
        print(f"Worked {worker_idx} testing program {pidx}. Queue size: {qsize}")
        Model = run_code_and_get_class(strip_ticks(program))
        score, n_examples, n_epochs = compute_accuracy(Model, args)
        result_queue.put((pidx, score, n_examples, n_epochs))
