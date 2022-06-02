# please fill this @neelabh
teachers = {
    "cifar10": {
        "resnet18" : [
            "checkpoint/cifar10/23-May_resnet18_focal_loss_gamma=1.0/model_best.pth", # calibrated
            "checkpoint/cifar10/24-May_resnet18_focal_loss_gamma=3.0/model_best.pth" # unCalibrated
        ],
        "resnet34" : [
            "checkpoint/cifar10/24-May_resnet34_focal_loss_gamma=3.0/model_best.pth", # least calibrated
            "checkpoint/cifar10/23-May_resnet34_FL+MDCA_gamma=2.0_beta=1.0/model_best.pth", # Calibrated
        ]
    },
    "cifar100" : {
        "resnet18" : [
            "checkpoint/cifar100/24-May_resnet18_cross_entropy/model_best.pth", # least calibrated
            "checkpoint/cifar100/24-May_resnet18_focal_loss_gamma=1.0/model_best.pth" # Calibrated
        ],
        "resnet34" : [
            "checkpoint/cifar100/24-May_resnet34_cross_entropy/model_best.pth", # least calibrated
            "checkpoint/cifar100/24-May_resnet34_FL+MDCA_gamma=2.0_beta=5.0/model_best.pth" # Calibrated
        ]
    }
}

students = ["convnet"] #, "resnet34"]
datasets = ["cifar10", "cifar100"]

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
                            # print(f"python train_student.py --dataset {dataset} --model {student} --teacher {teacher_model} --checkpoint {teacher_path} --temp {temp} --dw {dw} --lr 0.1 --lr-decay-factor 0.1 --wd 5e-4 --train-batch-size 128 --schedule-steps 100 150 --epochs 200 --exp_name runid={idx}")
                            print(f"python train_student.py --dataset {dataset} --model {student} --teacher {teacher_model} --checkpoint {teacher_path} --temp {temp} --dw {dw} --lr 0.001 --wd 1e-4 --train-batch-size 128 --epochs 200 --exp_name runid={idx}")
                            idx += 1                        
