import itertools
import os
import time
from subprocess import Popen, PIPE

# arch = os.environ.get("ARCH", "Resnet")
# num_gpus = int(os.environ.get("GPU_COUNT", 1))
# cpus_per_job = int(os.environ.get("USE_CPU", 12))
# arch = "mobile"
arch = "resnet"
jobs_per_gpu = 1
num_gpus = 4
num_jobs = num_gpus * jobs_per_gpu
num_cpus = 26
cpus_per_job = num_cpus // num_jobs
seeds = ["20260114"]

if arch.lower().startswith("mo"):
    script = "./scripts/experiment_mobilenet.sh"
elif arch.lower().startswith("res"):
    script = "./scripts/experiment_resnet.sh"
else:
    assert(False)

scripts = [script]

commands = [[] for _ in range(num_jobs)]
command_iter = itertools.cycle(range(len(commands)))

for seed in seeds:
    for script in scripts:
        augments = ["auto", "trivial", "augmix", "rand", "autoimg", "autosvhn", "none", "none"]
        cutmix = ["False"] * 7 + ["True"]


        for transform, cm, gpu_idx, command_idx in zip(augments, cutmix, itertools.cycle(range(num_gpus)), command_iter):
            command = [script, str(gpu_idx), str(cpus_per_job), transform, cm, seed]
            commands[command_idx].append(" ".join(command))

for command in commands:
    command = " ; ".join(["cd SQAKD/TinyImageNet/"] + command)
    print(command)
    
print("Done!")