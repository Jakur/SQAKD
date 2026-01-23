import itertools
import os
import time

num_gpus = int(os.environ.get("GPU_COUNT", 4))
script = "./scripts/run_cifar100_resnet32_EWGS+KDs.sh"


print(f"Number of GPUs: {num_gpus}")

jobs_per_gpu = 2

# augments = ["auto", "trivial", "augmix", "rand", "erasing", "autoimg", "autosvhn", "none"]
# if cutmix_status.lower().startswith("t"):
#     cutmix = ["True"]
# elif cutmix_status.lower().startswith("f"):
#     cutmix = ["False"]
# else:
#     cutmix = ["True", "False"]

method = ["at", "cc", "crd", "nst", "rkd", "sp", "rld", "kd"]
# method = ["sqakd", "at", "nst", "sp", "rkd", "crd", "fitnet", "cc", "vid", "fsp", "ft", "cktf"]
ours = ["t", "f"]

search = sorted(list(itertools.product(method, ours)))
print(search)

commands = [[] for _ in range(jobs_per_gpu * num_gpus)]

count = 0
for (gpu_idx, (meth, our)) in zip(itertools.cycle(range(num_gpus)), search):
    command = [script, str(gpu_idx), meth, our]
    value = f"{' '.join(command)}"
    commands[count % len(commands)].append(value)
    count += 1

for cmd_list in commands:
    cmd = " ; ".join(cmd_list)
    print(cmd)
    # print("\n")