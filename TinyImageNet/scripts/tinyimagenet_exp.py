import itertools
import os
import time
from subprocess import Popen, PIPE

arch = os.environ.get("ARCH", "Resnet")
num_gpus = int(os.environ.get("GPU_COUNT", 1))
cutmix_status = os.environ.get("USE_CUTMIX", "False")
cpus_per_job = int(os.environ.get("USE_CPU", 12))

if arch.lower().startswith("mo"):
    script = "./scripts/experiment_mobilenet.sh"
elif arch.lower().startswith("res"):
    script = "./scripts/experiment_resnet.sh"
else:
    assert(False)

print(f"Number of GPUs: {num_gpus}")

jobs_per_gpu = 1

augments = ["auto", "trivial", "augmix", "rand", "erasing", "autoimg", "autosvhn", "none"]
if cutmix_status.lower().startswith("t"):
    cutmix = ["True"]
elif cutmix_status.lower().startswith("f"):
    cutmix = ["False"]
else:
    cutmix = ["True", "False"]

search = sorted(list(itertools.product(augments, cutmix)))
print(search)

commands = [[] for _ in range(num_gpus)]

for (gpu_idx, (transform, cm)) in zip(itertools.cycle(range(num_gpus)), search):
    command = [script, str(gpu_idx), str(cpus_per_job), transform]
    value = f"{' '.join(command)}"
    commands[gpu_idx].append(value)

for cmd_list in commands:
    cmd = " ; ".join(cmd_list)
    print(cmd)
    print("\n")