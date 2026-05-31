import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import itertools

for seed, n, r, T in itertools.product(range(1000),[25,50,100,200,400],[.1,.3,.5,.7,.9],[1,5,25]):
    subprocess.run(f"python ./test-sq.py {seed} {n} {r} {T}", shell=True)
for seed, n, r, T in itertools.product(range(1000),[25,50,100,200,400],[.1,.3,.5,.7,.9],[1,5,25]):
    subprocess.run(f"python ./valid-sq.py {seed} {n} {r} {T}", shell=True)
for n, r, T in itertools.product([25,50,100,200,400],[.1,.3,.5,.7,.9],[1,5,25]):
    subprocess.run(f"python ./plot-sq.py 0 {n} {r} {T}", shell=True)
