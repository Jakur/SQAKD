import itertools
import os
import time
from subprocess import Popen, PIPE

resnet = False
if resnet:
    script = './scripts/experiment_cifar100_resnet32.sh'
else:
    script = './scripts/experiment_cifar100_vgg13.sh'

num_gpus = 8
print(f"Number of GPUs: {num_gpus}")
jobs_per_gpu = 2
num_jobs = num_gpus * jobs_per_gpu
num_cpus = 192
# use_cores = num_cpus // num_gpus
cpus_per_job = num_cpus // num_jobs

augments = ["auto", "trivial", "augmix", "rand", "autoimg", "autosvhn", "none", "none"]
cutmix = ["False"] * 7 + ["True"]

# search = sorted(list(itertools.product(augments, cutmix)))
# print(search)

commands = [[] for _ in range(num_gpus)]

for transform, cm, gpu_idx in zip(augments, cutmix, itertools.cycle(range(num_gpus))):
    command = [script, str(gpu_idx), str(cpus_per_job), transform, cm]
    commands[gpu_idx].append(" ".join(command))

for command in commands:
    command = " ; ".join(command)
    print(command)
    

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

print("Done!")