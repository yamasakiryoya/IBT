import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import itertools

for seed, r in itertools.product(range(1000),[.1,.3,.5,.7,.9]):
    subprocess.run(f"python ./rating-sq.py {seed} {r}", shell=True)
for seed, r in itertools.product(range(1000),[.1,.3,.5,.7,.9]):
    subprocess.run(f"python ./rating-nll.py {seed} {r}", shell=True)
for seed, r in itertools.product(range(1000),[.1,.3,.5,.7,.9]):
    subprocess.run(f"python ./rating-sq2.py {seed} {r}", shell=True)
for seed, r in itertools.product(range(1000),[.1,.3,.5,.7,.9]):
    subprocess.run(f"python ./rating-nll2.py {seed} {r}", shell=True)