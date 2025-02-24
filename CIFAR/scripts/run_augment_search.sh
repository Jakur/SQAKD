set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE

# FP
if [ $METHOD_TYPE == "augment10/" ]
then
    python3 augment_search.py --gpu_id '0' \
                    --dataset 'cifar10' \
                    --teacher_arch 'vgg8_bn_fp' \
                    --num_workers 8 \
                    --batch_size 100 \
                    --seed 20240913 \
                    --teacher_path './results/CIFAR10_VGG8/fp_cutmix/checkpoint/last_checkpoint.pth'

elif [ $METHOD_TYPE == "augment/" ]
then
    python3 augment_search.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --teacher_arch 'vgg13_bn_fp' \
                    --num_workers 8 \
                    --batch_size 100 \
                    --seed 20240913 \
                    --teacher_path './results/CIFAR100_VGG13/fp_cutmix/checkpoint/last_checkpoint.pth'

elif [ $METHOD_TYPE == "augment_resnet10/" ]
then
    python3 augment_search.py --gpu_id '0' \
                    --dataset 'cifar10' \
                    --teacher_arch 'resnet20_fp' \
                    --num_workers 8 \
                    --batch_size 100 \
                    --seed 20240913 \
                    --teacher_path './results/CIFAR10_ResNet20/fp_cutmix/checkpoint/last_checkpoint.pth'

elif [ $METHOD_TYPE == "augment_resnet/" ]
then
    python3 augment_search.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --teacher_arch 'resnet32_fp' \
                    --num_workers 8 \
                    --batch_size 100 \
                    --seed 20240913 \
                    --teacher_path './results/CIFAR100_ResNet32/fp_cutmix/checkpoint/last_checkpoint.pth'
fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"

