import itertools
import queue
import re
import dspy
import textwrap
import pydantic

import model_tester
from default_progs import template


def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def MySignature(args, personality):
    class BaseSignature(dspy.Signature):
        program: str = dspy.OutputField(
            desc="A python lightning class, called ImageModel, you can include imports, but no other code."
        )

        @pydantic.field_validator("program")
        def check_syntax(cls, v):
            v = model_tester.strip_ticks(v)
            lines = v.split("\n")
            allowed = ["import", "from", "class", " ", "#"]
            for line in lines:
                if line and not any(line.startswith(prefix) for prefix in allowed):
                    raise ValueError(f"Don't write any code besides the class. You wrote {repr(line)}")
            if "torchvision.models" in v:
                raise ValueError("Don't import torchvision.models")
            if "class ImageModel" not in v:
                raise ValueError("You must define one class named ImageModel")
            if "self.batch_size" not in v or "self.transform" not in v:
                raise ValueError(
                    "Remember to define self.batch_size and self.transform, such as self.batch_size=64 and self.transform=transforms.Compose([transforms.ToTensor()])"
                )
            try:
                # Attempt to compile the code snippet
                compile(v, "<string>", "exec")
                # Attempt to run the code snippet
                Model = model_tester.run_code_and_get_class(v)
                total_params, _ = get_model_parameters(Model())
                if total_params > 10**7:
                    raise ValueError(f"You used {total_params} parameters. Please keep it under 1,000,000")
                _ = model_tester.compute_accuracy(Model, args, test_run=True)
            except Exception as e:
                raise ValueError(f"Code did not run: {e}")
            return v

    class ImproveSignature(BaseSignature):
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
        analysis: str = dspy.OutputField(
            desc="Short analysis of what the previous models did that worked well or didn't work well. How can you improve on them? Answer in plain text, no Markdown"
        )
        explanation: str = dspy.OutputField(desc="Short explanation of what's different")

        @pydantic.field_validator("analysis")
        def check_for_score(cls, s):
            # Sometimes the model invents its own score. Remove that.
            return re.sub("Score: [\d\.]+", "", s)

    class InitialSignature(BaseSignature):
        f"""Write a simple python class for training a model for {args.dataset}. Use the template: ```python\n{template}\n```"""
        pass

    return ImproveSignature, InitialSignature


def model_producer(
    args,
    personality: str,
    model_queue: queue.Queue,
    demo_queue: queue.Queue,
    worker_idx,
) -> None:
    make_initial = worker_idx == 0

    ImproveSignature, InitialSignature = MySignature(args, personality)
    proposer = dspy.TypedPredictor(ImproveSignature, explain_errors=True, max_retries=10)

    used_demo_subsets = set()
    demos = {}
    while True:
        while not demo_queue.empty():
            pidx, demo = demo_queue.get()
            demos[pidx] = demo

        if not demos:
            if make_initial:
                print(f"Making initial program from {worker_idx}...")
                initial_proposer = dspy.TypedPredictor(InitialSignature, explain_errors=True, max_retries=10)
                try:
                    pred = initial_proposer()
                except ValueError as e:
                    print(f"Worked {worker_idx} failed: {e}")
                    dspy.settings.lm.inspect_history(n=1)
                    continue
                print(f"Success! {worker_idx}")
                model_queue.put((pred.program, "Baseline model."))
                make_initial = False
                continue

            # Wait for the first result
            pidx, demo = demo_queue.get()
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
            pidx, demo = demo_queue.get()
            demos[pidx] = demo
            continue

        proposer.predictor.demos = [demo for i, demo in subset]
        target_score = (max(demo.score for demo in demos.values()) + 1) / 2
        print(f"Making program from {worker_idx}...")
        try:
            pred = proposer(score=target_score)
        except ValueError as e:
            print(f"Worked {worker_idx} failed: {e}")
            dspy.settings.lm.inspect_history(n=1)
            continue
        print(f"Success! {worker_idx}")
        model_queue.put((pred.program, pred.analysis))
