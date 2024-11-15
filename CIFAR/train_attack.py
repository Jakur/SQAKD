from custom_models import *
from custom_modules import *
from utils import *
from distill import DistillKL, PKT
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchattacks import PGD
from torchmetrics import Accuracy

class Normalizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 

    def forward(self, x):
        mean = torch.tensor((0.4914, 0.4822, 0.4465))
        mean = torch.repeat_interleave(mean, 32 * 32).view(3, 32, 32).to(x.device)
        std = torch.tensor((0.2023, 0.1994, 0.2010))
        std = torch.repeat_interleave(std, 32 * 32).view(3, 32, 32).to(x.device)
        normalized_x = (x - mean) / std
        return self.model(normalized_x)

def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    # Most of this is dummy settings, but I need a bunch of the defaults
    parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (CIFAR)")
    # data and model
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), 
                                    help='dataset to use CIFAR10|CIFAR100')
    parser.add_argument('--arch', type=str, default='resnet20_quant', help='model architecture')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=None, help='seed for initialization')

    # training settings
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs for training')
    parser.add_argument('--optimizer_m', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for model paramters')
    parser.add_argument('--optimizer_q', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for quantizer paramters')
    parser.add_argument('--lr_m', type=float, default=1e-3, help='learning rate for model parameters')
    parser.add_argument('--lr_q', type=float, default=1e-5, help='learning rate for quantizer parameters')
    parser.add_argument('--lr_m_end', type=float, default=0.0, help='final learning rate for model parameters (for cosine)')
    parser.add_argument('--lr_q_end', type=float, default=0.0, help='final learning rate for quantizer parameters (for cosine)')
    parser.add_argument('--decay_schedule_m', type=str, default='150-300', help='learning rate decaying schedule (for step)')
    parser.add_argument('--decay_schedule_q', type=str, default='150-300', help='learning rate decaying schedule (for step)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for model parameters')
    parser.add_argument('--lr_scheduler_m', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
    parser.add_argument('--lr_scheduler_q', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')

    # arguments for distillation
    parser.add_argument('--distill', type=str2bool, default=True, help='do weight distillation')
    parser.add_argument('--distill_weight', type=float, default=0.5, help='Weight of distillation loss, existing loss will be (1-distill_weight)')

    # arguments for quantization
    parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='do weight quantization')
    parser.add_argument('--QActFlag', type=str2bool, default=True, help='do activation quantization')
    parser.add_argument('--weight_levels', type=int, default=2, help='number of weight quantization levels')
    parser.add_argument('--act_levels', type=int, default=2, help='number of activation quantization levels')
    parser.add_argument('--baseline', type=str2bool, default=False, help='training with STE')
    parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0, help='scaling factor for weights')
    parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0, help='scaling factor for activations')
    parser.add_argument('--use_hessian', type=str2bool, default=True, help='update scsaling factor using Hessian trace')
    parser.add_argument('--update_every', type=int, default=10, help='update interval in terms of epochs')

    # logging and misc
    parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
    parser.add_argument('--log_dir', type=str, default='../results/ResNet20_CIFAR10/W1A1/')
    parser.add_argument('--load_pretrain', type=str2bool, default=True, help='load pretrained full-precision model')
    parser.add_argument('--pretrain_path', type=str, default='../results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth', 
                                        help='path for pretrained full-preicion model')
    args = parser.parse_args()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        num_classes = 10
        args.num_classes = 10
        test_dataset = dsets.CIFAR10(root='../data/CIFAR10/',
                                train=False, 
                                transform=transform_test)
        vanilla_set = dsets.CIFAR10(root='../data/CIFAR10/',
                                train=False, 
                                transform=transforms.ToTensor())
    else:
        raise NotImplementedError
    
    arg_dict = vars(args)
    device = "cuda" 
    model_class_t = globals().get(args.teacher_arch)
    model_t = model_class_t(args)
    model_t.to(device)

    model_t = utils.load_teacher_model(model_t, args.teacher_path)
    trained_model = torch.load(args.pretrain_path)
    model_distill.load_state_dict(trained_model["model"])
    student = resnet20_quant(args).to(device)
    student_params = torch.load("foo_checkpoint.pth")
    # student_params = torch.load("../results/ResNet20_CIFAR10/ours(hess)/distill/checkpoint/last_checkpoint.pth")
    # student_params = torch.load("../results/ResNet20_CIFAR10/ours(hess)/distill5/checkpoint/last_checkpoint.pth")
    # student_params = torch.load('../results/ResNet20_CIFAR10/ours(hess)/W1A1/checkpoint/last_checkpoint.pth')
    student.load_state_dict(student_params["model"])
    


    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=100,
                                        shuffle=False,
                                        num_workers=args.num_workers)
    vanilla_loader = torch.utils.data.DataLoader(dataset=vanilla_set,
                                        batch_size=100,
                                        shuffle=False,
                                        num_workers=args.num_workers)
    
    def sanity_check():
        for (batch, other_batch) in zip(vanilla_loader, test_loader):
            x, _y = batch
            x = x.to(device)
            val_x, _ = other_batch
            val_x = val_x.to(device)
            # print(x[0, 0, :])
            mean = torch.tensor((0.4914, 0.4822, 0.4465))
            mean = torch.repeat_interleave(mean, 32 * 32).view(3, 32, 32).to(device)
            std = torch.tensor((0.2023, 0.1994, 0.2010))
            std = torch.repeat_interleave(std, 32 * 32).view(3, 32, 32).to(device)
            normalized_x = (x - mean) / std
            assert((normalized_x - val_x).sum() < 0.0001)
            break

    model = Normalizer(model_distill).eval()
    student = Normalizer(student).eval()
    attack = PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
    # attack = PGD(student, eps=8/255, alpha=1/255, steps=10, random_start=True)
    # labels obtained by mapping function as target labels.
    # shift all class loops one to the right, 1=>2, 2=>3, .., 9=>0
    attack.set_mode_targeted_by_function(target_map_function=lambda _images, labels:(labels+1)%num_classes)

    sanity_acc = Accuracy("multiclass", num_classes=num_classes).to(device)
    teacher_acc = Accuracy("multiclass", num_classes=num_classes).to(device)
    student_acc = Accuracy("multiclass", num_classes=num_classes).to(device)
    for batch in vanilla_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        goal_y = (y + 1) % num_classes
        adv_x = attack(x, y)
        y_teacher = model(adv_x)
        y_student = student(adv_x)
        teacher_acc(y_teacher, goal_y)
        student_acc(y_student, goal_y)
        with torch.no_grad():
            real_pred = attack.model(x)
            sanity_acc(real_pred, y)
    
    print("Transfer Learning scenario => Attack Teacher, Measure Impact on Student")
    print(f"Sanity Check: Accuracy Not Under Attack: {sanity_acc.compute()}")
    print(f"Teacher Attack Success Rate: {teacher_acc.compute()}")
    print(f"Student Attack Success Rate: {student_acc.compute()}")



    


if __name__ == "__main__":
    main()