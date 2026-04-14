from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def rld_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand, alpha_temperature):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    # scd loss
    student_gt_mask = _get_gt_mask(logits_student, target)
    student_other_mask = _get_other_mask(logits_student, target)
    max_index = torch.argmax(logits_teacher, dim=1)
    teacher_max_mask = _get_gt_mask(logits_teacher, max_index)
    teacher_other_mask = _get_other_mask(logits_teacher, max_index)
    pred_student = F.softmax(logits_student / alpha_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / alpha_temperature, dim=1)
    pred_student = cat_mask(pred_student, student_gt_mask, student_other_mask)
    pred_teacher = cat_mask(pred_teacher, teacher_max_mask, teacher_other_mask)
    log_pred_student = torch.log(pred_student)
    scd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (alpha_temperature**2)

    # mcd loss
    mask = _get_ge_mask(logits_teacher, target)
    assert mask.shape == logits_student.shape
    masked_student = (logits_student / temperature).masked_fill(mask, -1e9)
    log_pred_student_part2 = F.log_softmax(masked_student, dim=1)
    masked_teacher = (logits_teacher / temperature).masked_fill(mask, -1e9)
    pred_teacher_part2 = F.softmax(masked_teacher, dim=1)
    mcd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean') * (temperature**2)

    return alpha * scd_loss + beta * mcd_loss


def _get_ge_mask(logits, target):
    assert logits.dim() == 2 and target.dim() == 1 and logits.size(0) == target.size(0)
    gt_value = torch.gather(logits, 1, target.unsqueeze(1))
    mask = torch.where(logits >= gt_value, 1, 0).bool()
    return mask

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdim=True)
    t2 = (t * mask2).sum(dim=1, keepdim=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class RLD(nn.Module):

    def __init__(self, alpha=1.0, beta=8.0, alpha_temp=1.0, 
                 temp=4.0, warmup=20, logit_stand=False):
        super(RLD, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temp
        self.warmup = warmup
        self.logit_stand = logit_stand # Normalization, appears to be false in paper 
        self.alpha_temperature = alpha_temp

    def forward(self, logits_student, logits_teacher, target, epoch=1000):
        # logits_student, _ = self.student(image)
        # with torch.no_grad():
        #     logits_teacher, _ = self.teacher(image)

        # losses
        loss_rld = min(epoch / self.warmup, 1.0) * rld_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            self.logit_stand,
            self.alpha_temperature,
        )
        return loss_rld