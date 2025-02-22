import itertools
import os
import time
from subprocess import Popen, PIPE

num_gpus = os.environ.get("GPU_COUNT", 1)
# num_cpus = 64
# use_cores = num_cpus // num_gpus
num_jobs = 8

augments = ["auto", "trivial", "augmix", "rand", "erasing", "autoimg", "autosvhn", "none"]
cutmix = ["True", "False"]

search = sorted(list(itertools.product(augments, cutmix)))

jobs = [None for _ in range(num_gpus)]

def is_done():
    if len(search) > 0:
        return False
    return all([x is None for x in jobs])

while not is_done():
    retcode = 0
    for (gpu_idx, job) in enumerate(jobs):
        if job is None:
            (transform, cm) = search.pop()
            command = ['./scripts/experiment_cifar100_resnet32.sh', str(gpu_idx), str(num_jobs), transform, cm]
            process = Popen(command, stdout=PIPE, stderr=PIPE)
            print(f'Executing: {command}')
            jobs[gpu_idx] = process
            break
        else:
            retcode = job.poll()
            if retcode is not None: # Process finished.
                jobs[gpu_idx] = None
                break
            else: # No process is done, wait a bit and check again.
                time.sleep(0.1)

print("Done!")