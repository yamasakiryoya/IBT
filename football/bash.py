#python bash.py
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
rank = comm.Get_rank()  # 自分の番号 (0 ～ 1919)
size = comm.Get_size()  # 全プロセス数 (1920)


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
    code, seed, r = params
    if code==0: subprocess.run(f"python ./rating-sq.py {seed} {r}", shell=True)
    if code==1: subprocess.run(f"python ./rating-nll.py {seed} {r}", shell=True)


tasks = []
for j in range(100):
    tasks.extend(list(itertools.product([0],[j],[.1,.3,.5,.7,.9])))
    tasks.extend(list(itertools.product([1],[j],[.1,.3,.5,.7,.9])))
MP(run, tasks)