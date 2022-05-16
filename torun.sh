# try to train for all possible (model,dataset) pairs
# make a spreadsheet so that we keep storing metrics etc.
# we need exhaustive experiments, so thoda time lgega :P

# TODO
# if you can, maybe make a python script, that puts these on training, whenever any of the
# model finishes training @neelabh

# model = [resnet110, resnet152, resnet50]
# dataset = [cifar10, cifar100]

python train_teacher.py \
--dataset cifar100 \
--model resnet110 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss cross_entropy


# try to run on all configurations for (gamma, beta) pairs gamma in [1,2,3] beta in [1,5,10]
CUDA_VISIBLE_DEVICES=7 python train_teacher.py \
--dataset cifar100 \
--model resnet110 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss FL+MDCA \
--gamma 1.0 \
--beta 1.0

# try for all gamma values = [1, 2, 3]
CUDA_VISIBLE_DEVICES=7 python train_teacher.py \
--dataset cifar100 \
--model resnet110 \
--lr 0.1 \
--lr-decay-factor 0.1 \
--wd 5e-4 \
--train-batch-size 128 \
--schedule-steps 100 150 \
--epochs 200 \
--loss focal_loss \
--gamma 1.0



# Free gpus = [5, 6, 7]
# run the following commmand
simple_gpu_scheduler --gpus 5 6 7 < gpu_commands_hard.txt


# not so free  = [0, 1, 2, 3, 4, 5, 6, 7]
# run the following commmand
simple_gpu_scheduler --gpus 0 1 2 3 4 5 6 7 < gpu_commands_easy.txt