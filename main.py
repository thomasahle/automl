# import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import datetime
from multiprocessing import Queue
from pathlib import Path
import queue
import re
import select
import threading
import dspy
import torch
import random
import argparse
import dotenv
import os

from model_producer import model_producer, make_model, get_model_parameters
from model_tester import model_tester, run_code_and_get_class, strip_ticks
from model_eval import model_eval


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["mnist", "cifar10"])
parser.add_argument("train_time", type=int, default=3)
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
parser.add_argument(
    "--best-first",
    action="store_true",
    help="If set, the best programs will be at the top of the prompt. Otherwise, they will be at the bottom.",
)
args = parser.parse_args()


torch.set_float32_matmul_precision("medium")

personalities = [
    "Focus on exploitation: Make small changes to the hyper parameters of the best previous models to make them even better.",
    "Focus on exploration: Try out completely new approaches, that generate new knowledge. Don't be afraid to fail.",
    "Focus on speed: Make the model run faster, but don't sacrifice accuracy.",
    "Balance exploitation and exploration: Try out new approaches, but also make small changes to the best previous models.",
]


async def main():
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
    model_eval_to_worker = Queue()
    model_eval_from_worker = Queue()

    if len(personalities) < args.num_producers:
        min_length = args.num_producers - len(personalities)
        personalities.extend(
            dspy.TypedPredictor(
                f"personalities:list[str] -> more_personalities:Annotated[list[str], Field(min_length={min_length})]",
                f"Think of {min_length} more personalities.",
            )(personalities).more_personalities
        )

    examples = []
    tasks = set()
    executor = ThreadPoolExecutor(max_workers=args.num_producers + args.num_testers + args.num_evals)
    handlers = []

    for i, personality in zip(range(args.num_producers), personalities):
        handlers.append(ModelProducerHandler(args, examples, personality, i))

    for i in range(args.num_testers):
        tasks.add(executor.submit(model_tester, args, task_queue, result_queue, i))
    
    for handler in handlers:
        tasks.add(executor.submit(handler.start))

    # eval_threads = [
    #    threading.Thread(target=model_eval, args=(args, model_eval_to_worker, model_eval_from_worker, i))
    #    for i in range(args.num_evals)
    # ]
    # for t in producer_threads + tester_threads + eval_threads:
    #    t.start()

    while len(examples) < 100:
        finished, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            handler = task.result()
            tasks.add(executor.submit(handler.run))

    # TODO: Consider taking a folder of existing programs here to use as a seed
    # programs = []
    # examples = []
    # actual_scores = []
    # while len(programs) < 100:
    #     # Wait for one of the queues to become non-empty
    #     if all(q.empty() for q in demo_queues):
    #         print("Waiting for workers...")
    #         # TODO: Consdier spinning up more producer threads here...
    #     select.select([model_queue._reader, result_queue._reader, model_eval_from_worker._reader], [], [])

    #     # All processing in this loop is fast and non-blocking. The blocking / hard work
    #     # happens in the workers.
    #     for input_queue in [model_queue, result_queue, model_eval_from_worker]:
    #         try:
    #             value = input_queue.get(block=False)
    #         except queue.Empty:
    #             continue

    #         if input_queue is model_queue:
    #             model_queue_handler(task_queue, programs, value)

    #         elif input_queue is result_queue:
    #             result_queue_handler(output_folder, demo_queues, task_queue, programs, examples, actual_scores, value)

    #         elif input_queue is model_eval_from_worker:
    #             eval_queue_handler(model_eval_to_worker, value)


class ModelProducerHandler:
    def __init__(self, examples):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.output_folder = Path(f"best_programs_{current_time}")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.examples = examples
    
    async def start(self):
        pass

    async def handle(self, pidx, pred, score, n_examples, n_epochs):
        program = pred.program
        analysis = re.sub("Score: [\d\.]+", "", pred.analysis)
        print(f"Tested Program {pidx}")
        print(f"Actual score: {score:.3f}")
        print([actual_scores)
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
                analysis=analysis[:100],
                # plan=pred.plan[:300],
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
            print(f"Personalitity: {personalities[widx]}", file=f)
            print(f"Analysis:\n{analysis}\n", file=f)
            # print(f"Plan:\n{pred.plan}\n", file=f)
            print(f"Program:\n{program}", file=f)
            print(f"Explanation:\n{pred.explanation}\n", file=f)


def model_queue_handler(task_queue, programs, value):
    widx, pred = value
    program = pred.program
    # print("Anlysis:", analysis)
    pidx = len(programs)
    # print(f"Program ({pidx}):", program)
    task_queue.put((pidx, widx, program))
    programs.append(value)
    print(f"Added new program to queue. Queue size: {get_queue_size(task_queue)}")


class ModelEvalHandler:
    
def eval_queue_handler(model_eval_to_worker, value):
    # TODO: This is so annoying having to handle all those threads and queues.
    # Why can't we just have async / await?
    # Well, I guess we wanted to limit the amount of parallelism on the predictors...
    # Sure, but couldn't that still just be a thread pool?
    model_eval_to_worker.put(value)
    print(f"Added new program to eval queue. Queue size: {get_queue_size(model_eval_to_worker)}")


if __name__ == "__main__":
    main()
