import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import multiprocessing
from multiprocessing import Pool
import itertools
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def MP(run_script,tasks):
    if rank == 0:
        task_index = 0
        num_workers = size - 1
        closed_workers = 0
        for i in range(1, size):
            if task_index < len(tasks):
                comm.send(tasks[task_index], dest=i, tag=1)
                task_index += 1
            else:
                comm.send(None, dest=i, tag=0)
                closed_workers += 1
        while closed_workers < num_workers:
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            if task_index < len(tasks):
                comm.send(tasks[task_index], dest=source, tag=1)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=0)
                closed_workers += 1
    else:
        while True:
            status = MPI.Status()
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == 0: break
            run_script(task)
            comm.send(True, dest=0, tag=2)

def run(params):
    code, seed, n, r, T = params
    if code==0: subprocess.run(f"python ./rating-sq.py {seed} {n} {r} {T}", shell=True)
    if code==1: subprocess.run(f"python ./rating-nll.py {seed} {n} {r} {T}", shell=True)


tasks = []
for i in [25,50,100,200,400]:
    for j in range(100):
        tasks.extend(list(itertools.product([0],[j],[i],[.1,.3,.5,.7,.9],[1,5,25])))
        tasks.extend(list(itertools.product([1],[j],[i],[.1,.3,.5,.7,.9],[1,5,25])))
MP(run, tasks)