import multiprocessing
import os
import resource
import sys
import time
import traceback
from argparse import Namespace

import cifar_runner


# Computes the accuracy of the model in a separate process, with resource limits
# We have to just spin off a new process every time. Even though it would be nice
# to keep the process alive, it's not worth the effort.
def run_in_worker(code: str, args: Namespace, test_run=False, memory_limit_bytes=2**25):
    assert isinstance(code, str)

    read_stdout, write_stdout = multiprocessing.Pipe(duplex=False)
    read_stderr, write_stderr = multiprocessing.Pipe(duplex=False)
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(
        target=main_wrapper,
        args=(
            code,
            "cpu" if test_run else "cuda",
            args.dataset,
            test_run,
            args.train_time,
            args.n_runs,
            memory_limit_bytes,
            child_conn,
            write_stdout,
            write_stderr,
            args.verbose,
        ),
    )
    p.start()

    # We give the process some extra time to finish, since there is some overhead in starting the process
    start_time = time.time()
    timeout = args.train_time * (1 + args.n_runs) + args.train_overhead
    if args.verbose:
        print(f"Waiting for process to finish. Timeout: {timeout}")
    p.join(timeout)

    # If the process is still alive after time_limit_sec, terminate it
    if p.is_alive():
        if args.verbose:
            print("Process timeout. Killing process...")
        p.terminate()
        p.join()  # TODO: Do we really need to wait for the termination to finish?
        if args.verbose:
            print("Kill complete.")
    else:
        if args.verbose:
            print(f"Process finished naturally after {time.time() - start_time:.2}s.")

    # Get stdout and stderr. First close the write end of the pipes to flush the data.
    # Then read the data from the read end of the pipes.
    write_stdout.close()
    write_stderr.close()
    with os.fdopen(read_stdout.fileno()) as file:
        stdout = file.read()
    with os.fdopen(read_stderr.fileno()) as file:
        stderr = file.read()

    # The readers need to be cleaned up in a slightly messy way, because of the way we
    # messed with the underlying file descriptors
    for conn in [read_stdout, read_stderr]:
        try:
            conn.close()
        except OSError:
            pass

    # Check if the process returned any result. If not, presumably it was killed.
    # For some reason this works better if it's done after the stdout/stderr stuff.
    result = None
    try:
        if parent_conn.poll():
            result = parent_conn.recv()
    except FileNotFoundError as e:
        print(traceback.format_exc())
        if args.verbose:
            print(f"Error: {e}. (Timeout?)")
    if result is None:
        result = {
            "traceback": "",
            "error": TimeoutError("The process did not return any result."),
            "result": (0, 0),
        }

    result["stdout"] = stdout
    result["stderr"] = stderr

    # Even if we got a result, we still need to check if there was an error.
    if result["error"] is not None:
        if test_run:
            raise result["error"]
        if args.verbose:
            print(f"Warning: The process failed with error '{result['error']}'.")

    return result


class VerboseWrapper:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        if self.stream2 is not None:
            self.stream2.write(data)


def main_wrapper(
    code,
    device,
    dataset,
    test_run,
    time_limit,
    n_runs,
    memory_limit_bytes,
    child_conn,
    stdout_conn,
    stderr_conn,
    verbose,
):
    # Capture stdout and stderr
    sys.stdout = VerboseWrapper(os.fdopen(stdout_conn.fileno(), "w", buffering=1), sys.stdout if verbose else None)
    sys.stderr = VerboseWrapper(os.fdopen(stderr_conn.fileno(), "w", buffering=1), sys.stderr if verbose else None)

    # Try to limit the maximal memory usage. Though this is not guaranteed to work.
    # resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    try:
        result = cifar_runner.main(code, device, dataset, time_limit, test_run, compile=False, n_runs=n_runs)
    except Exception as e:
        trace = traceback.format_exc()
        error = str(e)
        result = (0, 0)
    else:
        trace = None
        error = None

    print(f"{result=}")
    child_conn.send(
        {
            # "result": (0, 0),
            "result": result,
        }
    )
    # child_conn.send(
    #     {
    #         "traceback": trace,
    #         "error": error,
    #         "result": (0, 0),
    #     }
    # )
    stdout_conn.close()
    stderr_conn.close()
