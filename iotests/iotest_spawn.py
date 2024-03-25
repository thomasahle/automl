import multiprocessing
import os
import sys
import time

def worker(path, child_conn):
    sys.stdout = open(path, "w")
    print("Started...")
    for i in range(10):
        time.sleep(.1)
        print(i)
    child_conn.send({"result": 10})

def main():
    read_stdout, write_stdout = os.pipe()
    path = f'/proc/{os.getpid()}/fd/{write_stdout}'
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(target=worker, args=(path, child_conn))
    p.start()
    p.join()

    os.close(write_stdout)
    with os.fdopen(read_stdout) as stdout:
        print("Got stdout:", stdout.read())

    print("got object:", parent_conn.recv())

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()

