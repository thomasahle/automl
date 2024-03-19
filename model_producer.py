import itertools
import queue
import re
import dspy
import textwrap
import pydantic

import model_tester
from default_progs import template
import default_progs


def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def check_program(args, v):
    v = model_tester.strip_ticks(v)
    lines = v.split("\n")
    allowed = ["import", "from", "class", " ", "#"]
    for line in lines:
        if line and not any(line.startswith(prefix) for prefix in allowed):
            raise ValueError(f"Don't write any code besides the class. You wrote {repr(line)}")
    if "torchvision.models" in v:
        raise ValueError("Don't import torchvision.models")
    if f"class {args.class_name}" not in v:
        raise ValueError(f"You must define one class named {args.class_name}")
    if "self.batch_size" not in v or "self.transform" not in v:
        raise ValueError(
            "Remember to define self.batch_size and self.transform, such as self.batch_size=64 and self.transform=transforms.Compose([transforms.ToTensor()])"
        )
    try:
        # Attempt to compile the code snippet
        compile(v, "<string>", "exec")
        # Attempt to run the code snippet
        Model = model_tester.run_code_and_get_class(v, args.class_name)
        total_params, _ = get_model_parameters(Model())
        if total_params > args.max_params:
            raise ValueError(f"You used {total_params:,} parameters. Please keep it under {args.max_params:,}")
        _ = model_tester.compute_accuracy(v, args, test_run=True)
    except Exception as e:
        raise ValueError(f"Code did not run: {e}")
    return v


def make_signatures(args, personality):
    class ImproveSignature(dspy.Signature):
        __doc__ = textwrap.dedent(f"""
        Write a new python lightning class to get the best score on {args.dataset} given
        {args.train_time} seconds training time. I will give you some examples, and you should
        use your creativity to suggest a better model, that will achieve higher accuracy than
        any of the previous models.

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

        {personality}
        """)

        score: float = dspy.InputField(desc="The accuracy the model should get")
        # score: float = dspy.OutputField(desc="The accuracy the model should get")
        analysis: str = dspy.OutputField(
            desc="Short analysis of what the previous models did that worked well or didn't work well. Answer in plain text, no Markdown"
        )
        plan: str = dspy.OutputField(desc="Based on the analysis, how will you design your new program?")
        program: str = dspy.OutputField(
            desc=f"A python lightning class, called {args.class_name}, you can include imports, but no other code."
        )
        explanation: str = dspy.OutputField(
            desc="Short explanation of how the program is different from the previous ones."
        )

        @pydantic.field_validator("program")
        def check_syntax(cls, v):
            return check_program(args, v)

        @pydantic.field_validator("analysis")
        def check_for_score(cls, s):
            # Sometimes the model invents its own score. Remove that.
            return re.sub("Score: [\d\.]+", "", s)

    class InitialSignature(dspy.Signature):
        f"""Write a simple python class for training a model for {args.dataset}. Use the template: ```python\n{template}\n```"""
        program: str = dspy.OutputField(
            desc=f"A python lightning class, called {args.class_name}, you can include imports, but no other code."
        )

        @pydantic.field_validator("program")
        def check_syntax(cls, v):
            return check_program(args, v)

    return ImproveSignature, InitialSignature


def model_producer(
    args,
    personality: str,
    model_queue: queue.Queue,
    demo_queue: queue.Queue,
    worker_idx,
) -> None:
    make_initial = worker_idx == 0

    ImproveSignature, InitialSignature = make_signatures(args, personality)
    proposer = dspy.TypedPredictor(ImproveSignature, explain_errors=True, max_retries=args.max_retries)

    used_demo_subsets = set()
    testing_backlog_size = 0
    demos = {}
    while True:
        while not demo_queue.empty():
            pidx, demo, testing_backlog_size = demo_queue.get()
            demos[pidx] = demo

        # If we are getting ahead of the testers, wait for them to catch up
        if testing_backlog_size > 10:
            print(f"Worked {worker_idx} waiting for testers to catch up...")
            pidx, demo, testing_backlog_size = demo_queue.get()
            demos[pidx] = demo

        if not demos:
            if make_initial:
                if args.from_scratch:
                    print(f"Making initial program from {worker_idx}...")
                    initial_proposer = dspy.TypedPredictor(
                        InitialSignature, explain_errors=True, max_retries=args.max_retries
                    )
                    try:
                        pred = initial_proposer()
                    except ValueError as e:
                        dspy.settings.lm.inspect_history(n=1)
                        print(f"Worked {worker_idx} failed: {e}")
                        continue
                    print(f"Success! {worker_idx}")
                    program = pred.program
                else:
                    if args.dataset == "mnist":
                        program = default_progs.mnist
                    elif args.dataset == "cifar10":
                        program = default_progs.cifar
                    else:
                        raise ValueError(f"Unsupported dataset: {args.dataset}")
                model_queue.put((worker_idx, dspy.Example(program=program, analysis="Baseline model.")))
                make_initial = False
                continue

            # Wait for the first result
            pidx, demo, testing_backlog_size = demo_queue.get()
            demos[pidx] = demo

        best = sorted(demos.items(), key=lambda x: x[1].score, reverse=True)
        # Find a subset we havne't tried yet
        max_examples = min(args.max_examples, len(best))
        for subset in itertools.combinations(best, max_examples):
            key = tuple(sorted(x[0] for x in subset))
            if key not in used_demo_subsets:
                used_demo_subsets.add(key)
                break
        else:
            # We've tried all subsets. Wait for a new demo.
            pidx, demo, testing_backlog_size = demo_queue.get()
            demos[pidx] = demo
            continue

        # Flip to keep the best at the bottom
        if not args.best_first:
            subset = subset[::-1]

        proposer.predictor.demos = [demo for i, demo in subset]
        target_score = (max(demo.score for demo in demos.values()) + 1) / 2
        print(f"Making program from {worker_idx}...")
        try:
            pred = proposer(score=target_score)
            # pred = proposer()
        except ValueError as e:
            dspy.settings.lm.inspect_history(n=1)
            print(f"Worked {worker_idx} failed: {e}")
            continue
        print(f"Success! {worker_idx}")
        model_queue.put((worker_idx, pred))
    print("Model producer stopped.")
