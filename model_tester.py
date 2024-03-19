import logging
import multiprocessing
import os
import sys
import time
import traceback
import pytorch_lightning as pl
from torchvision import transforms
from datetime import timedelta
import torch.nn.functional as F
from argparse import Namespace

from data import DataModule


class ExceptionInfo:
    def __init__(self, exc):
        self.exc = exc
        self.traceback = traceback.format_exc()

    def re_raise(self, verbose=True):
        if verbose:
            print(self.traceback)
        raise self.exc


# Computes the accuracy of the model in a separate process, with resource limits
def compute_accuracy(code: str, args: Namespace, test_run=False, memory_limit_bytes=2**25):
    result_queue = multiprocessing.Queue()
    assert isinstance(code, str)

    # Create a new process to run the compute_accuracy function with resource limits
    p = multiprocessing.Process(
        target=compute_accuracy_worker,
        args=(code, args, result_queue, memory_limit_bytes, test_run),
    )

    p.start()

    # We give the process some extra time to finish, since there is some overhead in starting the process
    p.join(args.train_time * 2 + 10)

    if p.is_alive():
        # If the process is still alive after time_limit_sec, terminate it
        p.terminate()
        p.join()

    if not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, ExceptionInfo):
            if test_run:
                result.re_raise(verbose=False)
            print(f"Warning: The process failed with excetion {result}.")
            return 0, 0, 0
        return result
    else:
        print("Warning: The process did not return any result.")
        return 0, 0, 0


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


# This is totally unsafe. Run at your own risk.
def run_code_and_get_class(code, class_name):
    namespace = {}
    exec(code, namespace)
    return namespace[class_name]


def strip_ticks(s):
    s = s.strip()
    if s.startswith("```python\n"):
        assert s.endswith("```")
        return s[10:-3]
    return s


def compute_accuracy_worker(
    code: str, args: Namespace, result_queue: multiprocessing.Queue, memory_limit_bytes: int, test_run=False
):
    # resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    try:
        result = compute_accuracy_inner(code, args, test_run)
    except Exception as e:
        result = ExceptionInfo(e)
    result_queue.put(result)


def compute_accuracy_inner(code: str, args: Namespace, test_run=False):
    Model = run_code_and_get_class(strip_ticks(code), args.class_name)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    if test_run:
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # This hides the CPU instruction set warnings (and info messages)

    # for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
    #    print(logger)

    if test_run:
        trainer = pl.Trainer(
            fast_dev_run=True,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            accelerator="cpu",
            devices=1,
        )
    else:
        # For counting the number of batches we train on
        class BatchCounterCallback(pl.Callback):
            def __init__(self):
                super().__init__()
                self.batch_count = 0

            def on_train_batch_end(self, *args):
                self.batch_count += 1

        batch_counter = BatchCounterCallback()
        trainer = pl.Trainer(
            max_time=timedelta(seconds=args.train_time),
            accelerator=args.accelerator,
            devices=1 if args.accelerator == "cpu" else args.devices,
            callbacks=[batch_counter],
            enable_checkpointing=False,
            enable_model_summary=True,
            precision="bf16-mixed",
        )

    # This initializes the model directly on gpu
    with trainer.init_module():
        model = Model()
    batch_size = getattr(model, "batch_size", 64)
    transform = getattr(model, "transform", transforms.Compose([transforms.ToTensor()]))

    print(f"Setting up data module {test_run=}")
    start = time.time()
    data_module = DataModule(batch_size=batch_size, transform=transform, dataset_name=args.dataset, test_run=test_run)
    try:
        # data_module.prepare_data()
        data_module.setup()
        print(f"Set up data module in {time.time() - start:.3f} seconds")
        if test_run:
            # We don't wrap this in a try-except block, because we want to see the error
            trainer.fit(model, datamodule=data_module)
            return None

        try:
            trainer.fit(model, datamodule=data_module)
            model.test_step = lambda *args: test_step(model, *args)  # Attach test_step
            res = trainer.test(model, dataloaders=data_module.test_dataloader(), verbose=False)
        except Exception as e:
            print(f"Warning: train/test gave exception {e}.")
            return 0, 0, 0

        n_examples = batch_counter.batch_count * batch_size
        fractional_epochs = batch_counter.batch_count / len(data_module.train_dataloader())
        return res[0]["test_acc"], n_examples, fractional_epochs
    finally:
        try:
            data_module.teardown("fit")
        except Exception as e:
            print(f"Warning: Could not teardown data module {e}")


def model_tester(args, task_queue, result_queue, worker_idx):
    while True:
        pidx, program = task_queue.get()
        try:
            qsize = task_queue.qsize()
        except NotImplementedError:
            qsize = -1  # Not implemented on MacOS
        print(f"Worker {worker_idx} testing program {pidx}. Queue size: {qsize}")
        score, n_examples, n_epochs = compute_accuracy(program, args)
        result_queue.put((pidx, score, n_examples, n_epochs))
    print("Model tester stopped.")
