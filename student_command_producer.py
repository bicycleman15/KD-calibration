# please fill this @neelabh
teachers = {
    "cifar10": {
        "resnet152" : [
            "checkpoint/cifar10/17-May_resnet152_focal_loss_gamma=3.0/model_best.pth", # least calibrated
            # "checkpoint/cifar10/16-May_resnet152_cross_entropy/model_best.pth", # unCalibrated
            "checkpoint/cifar10/16-May_resnet152_FL+MDCA_gamma=1.0_beta=10.0/model_best.pth" # Calibrated
        ],
        "resnet110" : [
            "checkpoint/cifar10/17-May_resnet110_focal_loss_gamma=3.0/model_best.pth", # least calibrated
            # "checkpoint/cifar10/16-May_resnet110_cross_entropy/model_best.pth", # unCalibrated
            "checkpoint/cifar10/16-May_resnet110_FL+MDCA_gamma=1.0_beta=10.0/model_best.pth", # Calibrated
        ]
    },
    "cifar100" : {
        "resnet152" : [
            "checkpoint/cifar100/18-May_resnet152_FL+MDCA_gamma=3.0_beta=1.0/model_best.pth", # least calibrated
            # "checkpoint/cifar100/17-May_resnet152_cross_entropy/model_best.pth", # unCalibrated
            "checkpoint/cifar100/17-May_resnet152_NLL+MDCA_beta=10.0/model_best.pth" # Calibrated
        ],
        "resnet110" : [
            "checkpoint/cifar100/18-May_resnet110_focal_loss_gamma=3.0/model_best.pth", # least calibrated
            # "checkpoint/cifar100/17-May_resnet110_cross_entropy/model_best.pth", # unCalibrated
            "checkpoint/cifar100/17-May_resnet110_NLL+MDCA_beta=5.0/model_best.pth" # Calibrated
        ]
    }
}

students = ["resnet18"] #, "resnet34"]
datasets = ["cifar10"] #, "cifar100"]

# please confirm this @neelabh
temps = [1, 1.5, 2, 3, 4.5, 6, 8, 10, 20]
dws = [0.99, 0.95, 0.5, 0.1, 0.05]

idx = 0

for dataset in datasets:
    for teacher_model in teachers[dataset]:
        for teacher_path in teachers[dataset][teacher_model]:
                for student in students:
                    for temp in temps:
                        for dw in dws:
                            print(f"python train_student.py --dataset {dataset} --model {student} --teacher {teacher_model} --checkpoint {teacher_path} --temp {temp} --dw {dw} --lr 0.1 --lr-decay-factor 0.1 --wd 5e-4 --train-batch-size 128 --schedule-steps 100 150 --epochs 200 --exp_name runid={idx}")
                            idx += 1                        
