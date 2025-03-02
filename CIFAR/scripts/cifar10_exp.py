import itertools
import os
import time
from subprocess import Popen, PIPE

resnet = os.environ.get("USE_RESNET", "True")
num_gpus = int(os.environ.get("GPU_COUNT", 1))
cutmix_status = os.environ.get("USE_CUTMIX", "True")
cpus_per_job = int(os.environ.get("USE_CPU", 10))

if resnet.lower().startswith("t"):
    script = "./scripts/experiment_cifar10_resnet20.sh"
else:
    script = './scripts/experiment_cifar10_vgg8.sh'

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
    command = [script, str(gpu_idx), str(cpus_per_job), transform, cm]
    value = f"{' '.join(command)}"
    commands[gpu_idx].append(value)

for cmd_list in commands:
    cmd = " ; ".join(cmd_list)
    print(cmd)
    print("\n")

# jobs = [None for _ in range(num_gpus * jobs_per_gpu)]

# def is_done():
#     if len(search) > 0:
#         return False
#     return all([x is None for x in jobs])

# while not is_done():
#     retcode = 0
#     for (idx, job) in enumerate(jobs):
#         if job is None:
#             if len(search) > 0:
#                 gpu_idx = idx // jobs_per_gpu
#                 (transform, cm) = search.pop()
#                 command = [script, str(gpu_idx), str(cpus_per_job), transform, cm]
#                 process = Popen(command, stdout=PIPE, stderr=PIPE)
#                 print(f'Executing: {command}')
#                 jobs[idx] = process
#                 break
#         else:
#             retcode = job.poll()
#             if retcode is not None: # Process finished.
#                 jobs[idx] = None
#                 break
#             else: # No process is done, wait a bit and check again.
#                 time.sleep(0.1)

# print("Done!")