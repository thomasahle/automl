import multiprocessing
import os
import resource
import sys
import traceback
from argparse import Namespace

import cifar_runner

# if sys.platform == "darwin":
# multiprocessing.set_start_method("fork")
multiprocessing.set_start_method("spawn")


# Computes the accuracy of the model in a separate process, with resource limits
# We have to just spin off a new process every time. Even though it would be nice
# to keep the process alive, it's not worth the effort.
def run_in_worker(code: str, args: Namespace, test_run=False, memory_limit_bytes=2**25):
    assert isinstance(code, str)

    read_stdout, write_stdout = os.pipe()
    read_stderr, write_stderr = os.pipe()
    # parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    q = multiprocessing.Queue()
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
            # child_conn,
            q,
            write_stdout,
            write_stderr,
        ),
    )
    p.start()

    # It is idiomatic for working with pipes to close unused ends.
    # child_conn.close() in the parent doesn't mean that child should close its end immediately
    # parent_conn.recv doesn't return as long as there is a chance that somebody will child_conn.send().
    # If child_conn is opened (in child or parent) then there is a chance
    # child_conn.close()

    # We give the process some extra time to finish, since there is some overhead in starting the process
    p.join(args.train_time * (1 + args.n_runs) + 10)

    if p.is_alive():
        # If the process is still alive after time_limit_sec, terminate it
        if args.verbose:
            print("Process timeout. Killing process...")
        p.terminate()
        p.join()
        if args.verbose:
            print("Kill complete.")

    result = q.get_nowait()
    # try:
    #     # Check if the process returned any result. If not, presumably it was killed.
    #     if not parent_conn.poll():
    #         result = {
    #             "traceback": "",
    #             "error": TimeoutError("The process did not return any result."),
    #             "result": (0, 0),
    #         }
    #     # Otherwise get normal result
    #     else:
    #         result = parent_conn.recv()
    # except FileNotFoundError as e:
    #     if args.verbose:
    #         print(f"Unable to read from parent_connection? {e}")
    #     result = {
    #         "traceback": traceback.format_exc(),
    #         "error": e,
    #         "result": (0, 0),
    #     }
    # finally:
    #     parent_conn.close()

    # Get stdout and stderr. First close the write end of the pipes to flush the data.
    # Then read the data from the read end of the pipes.
    os.close(write_stdout)
    os.close(write_stderr)
    with os.fdopen(read_stdout) as stdout:
        result["stdout"] = stdout.read()
    with os.fdopen(read_stderr) as stderr:
        result["stderr"] = stderr.read()

    # Even if we got a result, we still need to check if there was an error.
    if result["error"] is not None:
        if test_run:
            raise result["error"]
        if args.verbose:
            print(f"Warning: The process failed with excetion {result['error']}.")

    return result


def main_wrapper(
    code, device, dataset, test_run, time_limit, n_runs, memory_limit_bytes, child_conn, stdout_pipe, stderr_pipe
):
    # Capture stdout and stderr
    sys.stdout = os.fdopen(stdout_pipe, "w", buffering=1)
    sys.stderr = os.fdopen(stderr_pipe, "w", buffering=1)

    # Try to limit the maximal memory usage. Though this is not guaranteed to work.
    # resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    try:
        result = cifar_runner.main(code, device, dataset, time_limit, test_run, compile=False, n_runs=n_runs)
    except Exception as e:
        trace = traceback.format_exc()
        error = e
        result = (0, 0)
    else:
        trace = None
        error = None

    # child_conn.send(
    child_conn.put(
        {
            "traceback": trace,
            "error": error,
            "result": result,
        }
    )
    # child_conn.close()
