import itertools
import os
import time
from subprocess import Popen, PIPE

arch = os.environ.get("ARCH", "Mobilenet")
num_gpus = int(os.environ.get("GPU_COUNT", 1))
cutmix_status = os.environ.get("USE_CUTMIX", "False")
cpus_per_job = int(os.environ.get("USE_CPU", 12))

if arch.lower().startswith("mo"):
    script = "./scripts/experiment_cifar10_resnet20.sh"
else:
    assert(False)

print(f"Number of GPUs: {num_gpus}")

jobs_per_gpu = 1

augments = ["auto", "trivial", "augmix", "rand", "erasing", "autoimg", "autosvhn", "none"]
cutmix = ["False"]
# if cutmix_status.lower().startswith("t"):
#     cutmix = ["True"]
# elif cutmix_status.lower().startswith("f"):
#     cutmix = ["False"]
# else:
#     cutmix = ["True", "False"]

search = sorted(list(itertools.product(augments, cutmix)))
print(search)

jobs = [None for _ in range(num_gpus * jobs_per_gpu)]

def is_done():
    if len(search) > 0:
        return False
    return all([x is None for x in jobs])

while not is_done():
    retcode = 0
    for (idx, job) in enumerate(jobs):
        if job is None:
            if len(search) > 0:
                gpu_idx = idx // jobs_per_gpu
                (transform, cm) = search.pop()
                command = [script, str(gpu_idx), str(cpus_per_job), transform]
                process = Popen(command, stdout=PIPE, stderr=PIPE)
                print(f'Executing: {command}')
                jobs[idx] = process
                break
        else:
            retcode = job.poll()
            if retcode is not None: # Process finished.
                jobs[idx] = None
                break
            else: # No process is done, wait a bit and check again.
                time.sleep(0.1)

print("Done!")