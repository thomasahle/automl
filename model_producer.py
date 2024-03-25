import itertools
import re
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
        Write a new pytorch module to get the best score on {args.dataset} given
        {args.train_time} seconds training time. I will give you some examples, and you should
        use your creativity to suggest a better model, that will achieve higher accuracy than
        any of the previous models.

        You can try things such as:
        - Changing the batch size or learning rate
        - Trying diferent architectures
        - Using different activation functions
        - Using different optimizers
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
        - Note the program runs with dtype=bfloat16 and memory_format=torch.channels_last
        - Don't import pretrained models
        """)

        score: float = dspy.InputField(desc="The accuracy the model should get")
        personality: str = dspy.InputField(desc="A personality to guide the model design")

        analysis: str = dspy.OutputField(
            desc="Short analysis of what the previous models did that worked well or didn't work well. "
            + "Answer in plain text, no Markdown."
        )
        # plan: str = dspy.OutputField(desc="Based on the analysis, how will you design your new program?")
        program: str = dspy.OutputField(
            desc=f"A pytorch module, called {args.class_name}, you can include imports, but no other code."
        )
        evaluation: str = dspy.OutputField(
            desc="Short explanation of how the program is different from the previous ones."
        )

        @pydantic.field_validator("program")
        def check_syntax(cls, v):
            return check_program(args, v)

        # Sometimes the model invents its own score. Remove that.
        # @pydantic.field_validator("analysis", "plan", "evaluation")
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


def make_from_demos(args, personality, demos, used_demo_subsets):
    proposer = dspy.TypedPredictor(
        ImproveSignature(args),
        explain_errors=True,
        max_retries=args.max_retries,
    )

    best = sorted(enumerate(demos), key=lambda x: x[1].score, reverse=True)
    # Find a subset we havne't tried yet
    max_examples = min(args.max_examples, len(best))
    for subset in itertools.combinations(best, max_examples):
        key = tuple(sorted(x[0] for x in subset))
        if key not in used_demo_subsets:
            used_demo_subsets.add(key)
            break
    else:
        # We've tried all subsets. Wait for a new demo.
        return None

    # Flip to keep the best at the bottom
    if not args.best_first:
        subset = subset[::-1]

    proposer.predictor.demos = [demo for _i, demo in subset]

    for demo in proposer.predictor.demos:
        for name in ImproveSignature(args).fields.keys():
            if not hasattr(demo, name):
                raise ValueError(f"Demo is missing field {name}")

    assert len(proposer.predictor.demos) > 0

    target_score = (max(demo.score for demo in demos) + 1) / 2
    try:
        pred = proposer(score=target_score, personality=personality)
    except ValueError as e:
        dspy.settings.lm.inspect_history(n=1)
        print(f"Worker failed: {e}")
        return None

    pred.program = strip_ticks(pred.program)
    return dspy.Example(**pred, personality=personality)
