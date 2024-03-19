# import queue
import datetime
from multiprocessing import Queue
from pathlib import Path
import queue
import select
import threading
import dspy
import torch
import random
import argparse
import dotenv
import os

from model_producer import model_producer, get_model_parameters
from model_tester import model_tester, run_code_and_get_class, strip_ticks


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["mnist", "cifar10"])
parser.add_argument("train_time", type=int, default=3)
parser.add_argument("--devices", type=str)
parser.add_argument("--accelerator", type=str, default="cpu")
parser.add_argument("--num-producers", type=int, default=2)
parser.add_argument("--num-testers", type=int, default=1)
parser.add_argument("--max-examples", type=int, default=30)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--from-scratch", action="store_true", help="Whether to create the initial model from scratch.")
parser.add_argument("--max-retries", type=int, default=10)
parser.add_argument("--class-name", type=str, default="ImageModel")
parser.add_argument("--max-params", type=int, default=10**7)
parser.add_argument(
    "--best-first",
    action="store_true",
    help="If set, the best programs will be at the top of the prompt. Otherwise, they will be at the bottom.",
)
args = parser.parse_args()


torch.set_float32_matmul_precision("medium")


def main():
    random.seed(42)
    torch.manual_seed(42)
    dotenv.load_dotenv(os.path.expanduser("~/.env"))
    lm = dspy.OpenAI(model="gpt-4-turbo-preview", max_tokens=4000)
    # lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    dspy.settings.configure(lm=lm)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_folder = Path(f"best_programs_{current_time}")
    output_folder.mkdir(parents=True, exist_ok=True)

    model_queue = Queue()
    demo_queues = [Queue() for _ in range(args.num_producers)]
    task_queue = Queue()
    result_queue = Queue()

    personalities = [
        "Focus on exploitation: Make small changes to the hyper parameters of the best previous models to make them even better.",
        "Focus on exploration: Try out completely new approaches, that generate new knowledge. Don't be afraid to fail.",
        "Focus on speed: Make the model run faster, but don't sacrifice accuracy.",
        "Balance exploitation and exploration: Try out new approaches, but also make small changes to the best previous models.",
    ]
    if len(personalities) < args.num_producers:
        min_length = args.num_producers - len(personalities)
        personalities += dspy.TypedPredictor(
            f"personalities:list[str] -> more_personalities:Annotated[list[str], Field(min_length={min_length})]",
            f"Think of {min_length} more personalities.",
        )(personalities).more_personalities

    producer_threads = []
    for i, personality in zip(range(args.num_producers), personalities):
        t = threading.Thread(target=model_producer, args=(args, personality, model_queue, demo_queues[i], i))
        producer_threads.append(t)
        t.start()
    tester_threads = []
    for i in range(args.num_testers):
        t = threading.Thread(target=model_tester, args=(args, task_queue, result_queue, i))
        tester_threads.append(t)
        t.start()

    programs = []
    # TODO: Consider taking a folder of existing programs here to use as a seed
    examples = []
    actual_scores = []
    while len(programs) < 100:
        # Wait for one of the queues to become non-empty
        if all(q.empty() for q in demo_queues):
            print("Waiting for workers...")
            # TODO: Consdier spinning up more producer threads here...
        select.select([model_queue._reader, result_queue._reader], [], [])

        # All processing in this loop is fast and non-blocking. The blocking / hard work
        # happens in the workers.
        for input_queue in [model_queue, result_queue]:
            try:
                value = input_queue.get(block=False)
            except queue.Empty:
                continue

            if input_queue is model_queue:
                model_queue_handler(task_queue, programs, value)

            elif input_queue is result_queue:
                result_queue_handler(output_folder, demo_queues, task_queue, programs, examples, actual_scores, value)

    lm.inspect_history(n=2)

    print("Best Program:")
    for demo in examples:
        if demo.score == max(actual_scores):
            print(demo.program)
            break

    if args.plot:
        import matplotlib.pyplot as plt

        plt.plot(actual_scores)
        plt.show()


def get_queue_size(queue):
    try:
        qsize = queue.qsize()
    except NotImplementedError:
        qsize = -1  # Not implemented on MacOS
    return qsize


def result_queue_handler(output_folder, demo_queues, task_queue, programs, examples, actual_scores, value):
    pidx, score, n_examples, n_epochs = value
    program, analysis = programs[pidx]
    print(f"Tested Program {pidx}")
    actual_scores.append(score)
    print(f"Actual score: {score:.3f}")
    print(actual_scores)
    Model = run_code_and_get_class(strip_ticks(program), args.class_name)
    total_params, _ = get_model_parameters(Model())
    speed = n_examples / args.train_time
    explanation = (
        f"Accuracy: {score:.3f}. "
        + f"Model with {total_params:,} parameters. "
        + f"Speed: {speed:.3f} examples per second. "
        + f"Completed {n_epochs:.3f} epochs."
    )
    print(explanation)
    if score > 0:
        example = dspy.Example(
            program=program,
            analysis=analysis[:100] + "...",
            score=score,
            explanation=explanation,
        )
        examples.append(example)
        for demo_queue in demo_queues:
            demo_queue.put((pidx, example, get_queue_size(task_queue)))
    else:
        print(f"Program {pidx} failed, so not adding to the demo queue.")

    # Save the program, analysis, and score to a text file
    file_path = output_folder / f"{pidx}_{score:.3f}.txt"
    with file_path.open("w") as f:
        print(f"Dataset: {args.dataset}; Time limit: {args.train_time}s", file=f)
        print(f"Score: {score:.6f}\n", file=f)
        print(explanation, file=f)
        print(f"Analysis:\n{analysis}\n", file=f)
        print(f"Program:\n{program}", file=f)


def model_queue_handler(task_queue, programs, value):
    program, _analysis = value
    # print("Anlysis:", analysis)
    pidx = len(programs)
    # print(f"Program ({pidx}):", program)
    task_queue.put((pidx, program))
    programs.append(value)
    print(f"Added new program to queue. Queue size: {get_queue_size(task_queue)}")


if __name__ == "__main__":
    main()
