import torch
import torch.nn as nn

import logging

class SoftCrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds, soft_targets):
        logit_pred = torch.log_softmax(preds, dim=1)
        return torch.mean(torch.sum(-soft_targets * logit_pred, dim=1))

class VanillaKD(nn.Module):
    def __init__(self, temp=20.0, distil_weight=0.5) -> None:
        super().__init__()
        self.temp = temp
        self.distil_weight = distil_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.aux_loss_fn = SoftCrossEntropy()

        logging.info(f"Using Vanilla KD with: T={self.temp}, dw={self.distil_weight}")

    def forward(self, student_output, teacher_output, labels):
        soft_teacher_out = torch.softmax(teacher_output / self.temp, dim=1)
        soft_student_out = torch.softmax(student_output / self.temp, dim=1)

        loss = (1 - self.distil_weight) * self.cross_entropy(student_output, labels)
        loss += (self.distil_weight * self.temp * self.temp) * self.aux_loss_fn(
            soft_student_out, soft_teacher_out
        )
        return loss