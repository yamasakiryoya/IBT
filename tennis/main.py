import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import itertools

for seed, r in itertools.product(range(1000),[.1,.3,.5,.7,.9]):
    subprocess.run(f"python ./test-sq.py {seed} {r}", shell=True)
for seed, r in itertools.product(range(1000),[.1,.3,.5,.7,.9]):
    subprocess.run(f"python ./test-nll.py {seed} {r}", shell=True)
for seed, r in itertools.product(range(1000),[.1,.3,.5,.7,.9]):
    subprocess.run(f"python ./valid-sq.py {seed} {r}", shell=True)
for seed, r in itertools.product(range(1000),[.1,.3,.5,.7,.9]):
    subprocess.run(f"python ./valid-nll.py {seed} {r}", shell=True)
for r in [.1,.3,.5,.7,.9]:
    subprocess.run(f"python ./plot-sq.py 0 {r}", shell=True)
for r in [.1,.3,.5,.7,.9]:
    subprocess.run(f"python ./plot-nll.py 0 {r}", shell=True)