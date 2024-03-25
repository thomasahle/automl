import multiprocessing
import os
import sys
import time


def worker(stdout_conn, child_conn):
    sys.stdout = os.fdopen(stdout_conn.fileno(), "w", buffering=1)
    print("Started...")
    for i in range(10):
        time.sleep(0.1)
        print(i, end=" ")
        if i % 3 == 0:
            print(flush=True)
    child_conn.send({"result": 10})
    stdout_conn.close()


def main(timeout):
    print(f"\nTimeout: {timeout}")
    read_stdout, write_stdout = multiprocessing.Pipe(duplex=False)
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(target=worker, args=(write_stdout, child_conn))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        print("Timeout. Terminating.")
        p.terminate()
        p.join()

    write_stdout.close()
    with os.fdopen(read_stdout.fileno()) as stdout:
        res = stdout.read()
    print("Got stdout:", res, end="")

    try:
        read_stdout.close()
    except OSError:
        pass

    if not parent_conn.poll():
        print("No object. (Killed)")
    else:
        print("Got object:", parent_conn.recv())


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(timeout=1.1)
    main(timeout=0.5)
    time.sleep(10)
