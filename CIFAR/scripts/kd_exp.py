import itertools
import os
import time

num_gpus = int(os.environ.get("GPU_COUNT", 1))
script = "./scripts/run_cifar100_vgg13_EWGS+KDs.sh"


print(f"Number of GPUs: {num_gpus}")

jobs_per_gpu = 1

# augments = ["auto", "trivial", "augmix", "rand", "erasing", "autoimg", "autosvhn", "none"]
# if cutmix_status.lower().startswith("t"):
#     cutmix = ["True"]
# elif cutmix_status.lower().startswith("f"):
#     cutmix = ["False"]
# else:
#     cutmix = ["True", "False"]

method = ["sqakd", "at", "nst", "sp", "rkd", "crd", "fitnet", "cc", "vid", "fsp", "ft", "cktf"]
ours = ["t", "f"]

search = sorted(list(itertools.product(method[0:4], ours)))
print(search)

commands = [[] for _ in range(num_gpus)]

for (gpu_idx, (meth, our)) in zip(itertools.cycle(range(num_gpus)), search):
    command = [script, str(gpu_idx), meth, our]
    value = f"{' '.join(command)}"
    commands[gpu_idx].append(value)

for cmd_list in commands:
    cmd = " ; ".join(cmd_list)
    print(cmd)
    print("\n")