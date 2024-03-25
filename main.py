from concurrent.futures import ThreadPoolExecutor
import datetime
from multiprocessing import Queue
import multiprocessing
from pathlib import Path
import threading
import dspy
import torch
import random
import argparse
import dotenv
import os

import model_producer
import model_tester2


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["mnist", "cifar10"])
parser.add_argument("train_time", type=int, default=5)
parser.add_argument("--train-overhead", type=int, default=20, help="Extra time to wait for training program to finish.")
parser.add_argument("--devices", type=str)
parser.add_argument("--accelerator", type=str, default="cpu")
parser.add_argument("--num-producers", type=int, default=2)
parser.add_argument("--num-testers", type=int, default=1)
parser.add_argument("--num-evals", type=int, default=2)
parser.add_argument("--max-examples", type=int, default=30)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--from-scratch", action="store_true", help="Whether to create the initial model from scratch.")
parser.add_argument("--max-retries", type=int, default=10)
parser.add_argument("--class-name", type=str, default="ImageModel")
parser.add_argument("--max-params", type=int, default=10**7)
parser.add_argument("--n-runs", type=int, default=3, help="Number of runs to average over.")
parser.add_argument(
    "--best-first",
    action="store_true",
    help="If set, the best programs will be at the top of the prompt. Otherwise, they will be at the bottom.",
)


personalities = [
    "Focus on exploitation: Make small changes to the hyper parameters of the best previous models to make them even better.",
    "Focus on exploration: Try out completely new approaches, that generate new knowledge. Don't be afraid to fail.",
    "Focus on speed: Make the model run faster, but don't sacrifice accuracy.",
    "Balance exploitation and exploration: Try out new approaches, but also make small changes to the best previous models.",
]


def main():
    random.seed(42)
    torch.manual_seed(42)
    dotenv.load_dotenv(os.path.expanduser("~/.env"))
    lm = dspy.OpenAI(model="gpt-4-turbo-preview", max_tokens=4000)
    # lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    dspy.settings.configure(lm=lm)
    args = parser.parse_args()

    # Ensure we have enough personalities
    if len(personalities) < args.num_producers:
        min_length = args.num_producers - len(personalities)
        personalities.extend(
            dspy.TypedPredictor(
                f"personalities:list[str] -> more_personalities:Annotated[list[str], Field(min_length={min_length})]",
                f"Think of {min_length} more personalities.",
            )(personalities).more_personalities
        )

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_folder = Path(f"best_programs_{current_time}")
    output_folder.mkdir(parents=True, exist_ok=True)

    demo_queues = [Queue() for _ in range(args.num_producers)]
    program_queue = Queue(maxsize=10)  # Limit the number of programs we can produce without testing any
    eval_queue = Queue()

    # Each thread/worker has an input queue and a list of output queues
    threads = [
        threading.Thread(target=ModelProducerWorker(i, args, personalities[i], demo_queues[i], program_queue).run)
        for i in range(args.num_evals)
    ]
    threads.append(
        threading.Thread(target=ModelEvalWorker(len(threads), args, program_queue, [eval_queue] + demo_queues).run)
    )
    for thread in threads:
        thread.start()

    # We could have a thread taking care of the eval_queue, but it's useful to do some work in
    # the main thread, so we can keep track of how many programs we've evaluated so far.
    for pidx in range(100):
        example = eval_queue.get()
        write_example_to_file(pidx, example, output_folder)


def write_example_to_file(pidx, example, args, output_folder):
    score = example.accuracy
    # Save the program, analysis, and score to a text file
    file_path = output_folder / f"{pidx}_{score:.3f}.txt"
    with file_path.open("w") as f:
        print(f"Dataset: {args.dataset}; Time limit: {args.train_time}s", file=f)
        print(f"Score: {score:.6f}\n", file=f)
        print(f"Personalitity: {example.personality}", file=f)
        print(f"Analysis:\n{example.analysis}\n", file=f)
        print(f"Program:\n{example.program}", file=f)
        print(f"Stdout:\n{example.stdout}\n", file=f)
        print(f"Evaluation:\n{example.evaluation}\n", file=f)


class ModelProducerWorker:
    def __init__(self, widx, args, personality, demo_queue, program_queue):
        self.widx = widx
        self.args = args
        self.personality = personality
        self.demo_queue = demo_queue
        self.program_queue = program_queue

    def run(self):
        demos = {}
        used_demo_subsets = set()
        make_initial = True
        while True:
            # Take everything from the queue
            while not self.demo_queue.empty():
                pidx, demo = self.demo_queue.get()
                demos[pidx] = demo

            # If we haven't received any demos, we may need to wait for them
            if not demos:
                if make_initial and (self.args.from_scratch or self.widx <= 1):
                    make_initial = False  # Only make the initial program once
                    program = model_producer.make_initial_program(self.args, self.widx)
                    if program is None:
                        print("Failed to make initial program.")
                        continue
                    self.program_queue.put(program)
                else:
                    print(f"Producer Worker {self.widx} waiting for demos...")
                    pidx, demo = self.demo_queue.get()
                    demos[pidx] = demo
                continue

            print(f"Making program from {self.widx}...")
            program = model_producer.make_from_demos(self.args, self.personality, demos, used_demo_subsets)
            if program is None:
                print(f"Worker {self.widx} failed to make program.")
                continue
            program.personality = self.personality

            # TODO: Maybe print something to the console indicating if we are going to sleep
            # while trying to put something in a full queue?
            self.program_queue.put(program)


class ModelEvalWorker:
    def __init__(self, widx, args, program_queue, output_queues):
        self.widx = widx
        self.args = args
        self.program_queue = program_queue
        self.output_queues = output_queues

    def run(self):
        with ThreadPoolExecutor(max_workers=5) as executor:
            while True:
                if self.program_queue.empty() and self.args.verbose:
                    print(f"Worker {self.widx} waiting for programs...")
                program = self.program_queue.get()
                # The point is that `run_in_worker` will block and ensure only one gpu-bound
                # process is running at a time.
                if self.args.verbose:
                    print(f"Worker {self.widx} got program, {program.program[:500]}...")
                result = model_tester2.run_in_worker(program.program, self.args, test_run=False)
                # But we can still run the evaluation in parallel, and definitely don't need to
                # wait for it to finish, before starting the next program on the gpu.
                executor.submit(self.post_process, program, result)

    def post_process(self, program, result):
        acc, std = result["result"]
        if acc == std == 0:
            print(f"Worker {self.widx} failed to evaluate program.")
            if self.args.verbose:
                print("Stdout:", result["stdout"])
                print("Stderr:", result["stderr"])
                print("Traceback:", result["traceback"])
            return

        print("Worker", self.widx, "evaluated program with accuracy", acc, "+/-", std)
        print("Test dspy")
        answer = dspy.TypedPredictor("question -> answer")("What is 1+1?").answer
        print(f"Answer: {answer}")
        print("Asking model to evaluate...")
        thoughts = dspy.TypedPredictor(
            "plan, program, stdout -> thoughts",
            (
                f"The following program achieved an accuracy of {acc:.3f} +/- {std:.3}."
                + "Describe what the program did well, and what it could have done better."
            ),
        )(program.analysis, program.program, result["stdout"]).thoughts
        print("Evaluation done:", thoughts)
        print("Sending to output queues...")
        for queue in self.output_queues:
            queue.put(
                dspy.Example(
                    analysis=program.analysis,
                    program=program.program,
                    personality=program.personality,
                    evaluation=thoughts,
                    stdout=result["stdout"],
                    accuracy=acc,
                    std=std,
                )
            )


if __name__ == "__main__":
    # "fork" doesn't work with cuda.
    multiprocessing.set_start_method("spawn")
    main()
