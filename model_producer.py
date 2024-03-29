import itertools
import re

import numpy as np
import dspy
import textwrap
import pydantic

import model_tester2
import cifar_runner


def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


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


def check_program(args, v):
    v = strip_ticks(v)
    lines = v.split("\n")
    allowed = ["import", "from", "class", " ", "#"]
    for line in lines:
        if line and not any(line.startswith(prefix) for prefix in allowed):
            raise ValueError(f"Don't write any code besides the class. You wrote {repr(line)}")
    # TODO: Maybe just limit imports to torch and torch.nn?
    if "torchvision.models" in v:
        raise ValueError("Don't import torchvision.models")
    if f"class {args.class_name}" not in v:
        raise ValueError(f"You must define one class named {args.class_name}")
    if "def get_optimizers(self):" not in v:
        raise ValueError("Remember to define a `def get_optimizers(self)` method in Net")
    try:
        # Attempt to compile the code snippet
        compile(v, "<string>", "exec")
        # Attempt to run the code snippet
        Model = run_code_and_get_class(v, args.class_name)
        total_params, _ = get_model_parameters(Model())
        if total_params > args.max_params:
            raise ValueError(f"You used {total_params:,} parameters. Please keep it under {args.max_params:,}")
        _ = model_tester2.run_in_worker(v, args, test_run=True)
    except Exception as e:
        raise ValueError(f"Code did not run: {e}")
    return v


def ImproveSignature(args):
    class ImproveSignature(dspy.Signature):
        __doc__ = textwrap.dedent(f"""
        Write a new PyTorch module to achieve the best score on the {args.dataset} dataset
        within {args.train_time} seconds of training time. Utilize the provided examples and
        your creativity to propose a better model that surpasses the accuracy of previous models.

        Consider experimenting with the following techniques and hyperparameters:
        - Batch size and learning rate adjustments
        - Activation functions (e.g., ReLU, LeakyReLU, Swish, etc.)
        - Optimizers (e.g., SGD, Adam, AdamW, etc.)
        - Normalization techniques (e.g., BatchNorm, LayerNorm, GroupNorm, etc.)
        - Loss functions (e.g., CrossEntropyLoss, FocalLoss, LabelSmoothingLoss, etc.)
        - Learning rate schedules (e.g., StepLR, CosineAnnealingLR, OneCycleLR, Custom schedules, etc.)
        - Weight initialization methods (e.g., Xavier, Kaiming, Orthogonal, SVD, etc.)
        - Regularization techniques (e.g., L1/L2 regularization, weight decay, dropout, etc.)
        - Layer types and sizes (e.g., Conv2d, Linear, ResNet blocks, Transformers, etc.)
        - Hyperparameter tuning (e.g., momentum, weight decay, dropout rates, kernel sizes, strides, etc.)
        - Model optimization for faster inference

        Ensure that your model:
        - Efficiently manages memory usage to avoid exceeding available resources
        - Generates unique model architectures for each new program
        - Properly handles the dtype (bfloat16) and memory format (torch.channels_last). (Consider using x.resize or nn.Flatten instead of x.view for tensor reshaping)
        - Avoids importing pretrained models to maintain originality

        By incorporating these suggestions and techniques, strive to develop a novel and high-performing PyTorch model tailored to the {args.dataset} dataset within the given training time constraint.
        Pay particular attention to the personality provided to guide your model design.
        """)

        score: float = dspy.InputField(
            desc="The target accuracy for the model to achieve on the {args.dataset} dataset"
        )
        personality: str = dspy.InputField(
            desc="A personality or design philosophy to guide the model architecture and hyperparameter choices"
        )
        analysis: str = dspy.OutputField(
            desc="Provide a concise analysis of the strengths and weaknesses of the previous models. "
            + "Based on this analysis, outline the approach you plan to take in designing the new model. "
            + "Answer in plain text without Markdown formatting."
        )
        program: str = dspy.OutputField(
            desc=f"Implement a PyTorch module named {args.class_name} that incorporates the insights from the analysis. "
            + "Include necessary imports within the module code, but do not include any additional code outside the module definition."
        )
        evaluation: str = dspy.OutputField(desc="Briefly explain how the proposed model differs from the previous ones")

        @pydantic.field_validator("program")
        def check_syntax(cls, v):
            return check_program(args, v)

        # Sometimes the model invents its own score. Remove that.
        @pydantic.field_validator("analysis", "evaluation")
        def check_for_score(cls, s):
            # Actually this doesn't do anything right now in dspy
            return re.sub("Score: [\d\.]+", "", s)

    return ImproveSignature


initial_template = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        ...

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(gamma=0.9)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        batch_size = 256
        return optimizer, scheduler, loss_fn, batch_size
"""


def InitialSignature(args):
    class InitialSignature(dspy.Signature):
        f"""Write a simple python class for training a model for {args.dataset}. Use the template: ```python\n{initial_template}\n```"""
        program: str = dspy.OutputField(desc="A pytorch module, `class Net(nn.Module)`. No other code.")

        @pydantic.field_validator("program")
        def check_syntax(cls, v):
            return check_program(args, v)

    return InitialSignature


def make_initial_program(args, i):
    if args.from_scratch:
        print("Making initial program...")
        initial_proposer = dspy.TypedPredictor(
            InitialSignature(args), explain_errors=True, max_retries=args.max_retries
        )
        try:
            pred = initial_proposer()
        except ValueError as e:
            dspy.settings.lm.inspect_history(n=1)
            print(f"Worker failed: {e}")
            return None
        print("Success!")
        program = pred.program

    else:
        # TODO: Maybe better to move this to main.py, since it'll be better able to
        # make decisions about how many initial programs to use.
        if args.dataset == "mnist":
            raise NotImplementedError("MNIST not supported yet")
        elif args.dataset == "cifar10":
            program = cifar_runner.sample_nets[i]
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    return dspy.Example(
        analysis="No previous models to analyze",
        # plan="Use a simple model to get a baseline accuracy.",
        program=program,
        evaluation="This is the first model.",
    )


def sample_key(demos, k, unused_keys, max_attempts=10, power=1):
    """Sample a key from the demos based on their scores."""
    weights = np.array([1 / r**power for r in range(1, len(demos) + 1)])
    ranked_demos = sorted(enumerate(demos), key=lambda x: x[1].score, reverse=True)
    for _ in range(max_attempts):
        selected_indices = np.random.choice(len(demos), size=k, replace=False, p=weights / np.sum(weights))
        key = tuple(sorted(ranked_demos[i][0] for i in selected_indices))
        if key not in unused_keys:
            return key
    return None


def find_unused_key(demos, k, unused_keys):
    """Try all length k combinations, starting from the lexicographically best."""
    best = sorted(enumerate(demos), key=lambda x: x[1].score, reverse=True)
    for subset in itertools.combinations(best, k):
        key = tuple(sorted(x[0] for x in subset))
        if key not in unused_keys:
            return key
    return None


def make_from_demos(args, personality, demos, used_demo_subsets):
    max_examples = min(args.max_examples, len(demos))

    # TODO: We are currently sampling independently (except without replacement) based on the score.
    # But we probably also want to do something to increase diversity in the samples, so we don't
    # just sample a bunch of small variations of the best model.
    key = sample_key(demos, max_examples, used_demo_subsets)
    if key is None:
        key = find_unused_key(demos, max_examples, used_demo_subsets)
        if key is None:
            print("We've tried all subsets. Wait for a new demo.")
            return None

    subset = [demos[i] for i in key]
    subset.sort(key=lambda x: x.score, reverse=args.best_first)

    proposer = dspy.TypedPredictor(
        ImproveSignature(args),
        explain_errors=True,
        max_retries=args.max_retries,
    )
    proposer.predictor.demos = subset

    # Validate demos
    for demo in proposer.predictor.demos:
        for name in ImproveSignature(args).fields.keys():
            if not hasattr(demo, name):
                raise ValueError(f"Demo is missing field {name}")
    assert len(proposer.predictor.demos) > 0

    # Prepare and process prediction
    target_score = (max(demo.score for demo in subset) + 1) / 2
    try:
        pred = proposer(score=target_score, personality=personality)
    except ValueError as e:
        dspy.settings.lm.inspect_history(n=1)
        print(f"Worker failed: {e}")
        return None

    pred.analysis = re.sub("Score: [\d\.]+", "", pred.analysis)
    pred.program = strip_ticks(pred.program)
    return dspy.Example(**pred, personality=personality)
