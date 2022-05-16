# Train cifar10
# Just replace resnet20 with other model names such as resnet18, resnet50, resnet110 to train on them
# you can also tweak hyper-parameters, look at utils/argparser.py for more parameters.

# teacher training on CIFAR10
CUDA_VISIBLE_DEVICES=6 python train_teacher.py \
--dataset cifar100 \
--model resnet50 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss mdca \
--gamma 3.0 \
--beta 10.0

CUDA_VISIBLE_DEVICES=6 python train_teacher.py \
--dataset cifar100 \
--model resnet50 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss cross_entropy

# loss = [cross_entropy, mdca]

CUDA_VISIBLE_DEVICES=4 python train_student.py \
--dataset cifar100 \
--model resnet20 \
--teacher resnet56 \
--checkpoint checkpoint/cifar100_pretrained/resnet56_cross_entropy.pth \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--exp_name ce_teacher