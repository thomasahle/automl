# import queue
from multiprocessing import Queue
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
parser.add_argument("dataset", choices=["mnist", "cifar"])
parser.add_argument("train_time", type=int, default=3)
parser.add_argument("--devices", type=str)
parser.add_argument("--accelerator", type=str, default="cpu")
parser.add_argument("--num-producers", type=int, default=2)
parser.add_argument("--num-testers", type=int, default=1)
parser.add_argument("--max-examples", type=int, default=30)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()


torch.set_float32_matmul_precision("medium")


def main():
    random.seed(42)
    torch.manual_seed(42)
    dotenv.load_dotenv(os.path.expanduser("~/.env"))
    lm = dspy.OpenAI(model="gpt-4-turbo-preview", max_tokens=4000)
    # lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    dspy.settings.configure(lm=lm)

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
        t = threading.Thread(
            target=model_producer,
            args=(
                args,
                personality,
                model_queue,
                demo_queues[i],
                i,
            ),
        )
        producer_threads.append(t)
        t.start()
    tester_threads = []
    for i in range(args.num_testers):
        t = threading.Thread(
            target=model_tester,
            args=(
                args,
                task_queue,
                result_queue,
                i,
            ),
        )
        tester_threads.append(t)
        t.start()

    programs = []
    examples = []
    actual_scores = []
    while len(programs) < 100:
        # Wait for one of the queues to become non-empty
        print("Waiting for workers...")
        select.select([model_queue._reader, result_queue._reader], [], [])

        for input_queue in [model_queue, result_queue]:
            try:
                value = input_queue.get(block=False)
            except queue.Empty:
                continue

            if input_queue is model_queue:
                program, analysis = value
                print("Anlysis:", analysis)
                pidx = len(programs)
                print(f"Program ({pidx}):", program)
                task_queue.put((pidx, program))
                programs.append(value)

            elif input_queue is result_queue:
                pidx, score, n_examples, n_epochs = value
                program, analysis = programs[pidx]
                Model = run_code_and_get_class(strip_ticks(program))
                print(f"Tested Program {pidx}")
                actual_scores.append(score)
                print(f"Actual score: {score:.2f}")
                print(actual_scores)
                speed = n_examples / args.train_time
                speed_text = f"Speed: {speed:.2f} examples per second. Completed {n_epochs:.2f} epochs."
                print(speed_text)
                total_params, _ = get_model_parameters(Model())
                print("Total parameters:", total_params)
                example = dspy.Example(
                    program=program,
                    analysis=analysis[:100] + "...",
                    score=score,
                    explanation=f"Model with {total_params} parameters. {speed_text}",
                )
                examples.append(example)
                for demo_queue in demo_queues:
                    demo_queue.put((pidx, example))

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


if __name__ == "__main__":
    main()
