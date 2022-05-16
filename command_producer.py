models = ["resnet110", "resnet152"]
datasets = ["cifar10", "cifar100"]
gammas = [1.0, 2.0, 3.0]
betas = [1.0, 5.0, 10.0]

for model in models:
    for dataset in datasets:
        # print(f"python train_teacher.py --dataset {dataset} --model {model} --lr 0.1 --lr-decay-factor 0.1 --wd 5e-4 --train-batch-size 128 --schedule-steps 100 150 --epochs 200 --loss cross_entropy")
        # for gamma in gammas:
        #     print(f"python train_teacher.py --dataset {dataset} --model {model} --lr 0.1 --lr-decay-factor 0.1 --wd 5e-4 --train-batch-size 128 --schedule-steps 100 150 --epochs 200 --loss focal_loss --gamma {gamma}")
        #     for beta in betas:
        #         print(f"python train_teacher.py --dataset {dataset} --model {model} --lr 0.1 --lr-decay-factor 0.1 --wd 5e-4 --train-batch-size 128 --schedule-steps 100 150 --epochs 200 --loss FL+MDCA --gamma {gamma} --beta {beta}")
        
        for beta in betas:
                print(f"python train_teacher.py --dataset {dataset} --model {model} --lr 0.1 --lr-decay-factor 0.1 --wd 5e-4 --train-batch-size 128 --schedule-steps 100 150 --epochs 200 --loss NLL+MDCA --beta {beta}")
