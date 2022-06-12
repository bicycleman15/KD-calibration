import os
import pandas as pd

import torch
import torch.optim as optim

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path

from solvers.runners import train, test
from solvers.loss import loss_dict

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from time import localtime, strftime

import logging
from utils import Logger

if __name__ == "__main__":

    # input size does not change
    torch.backends.cudnn.benchmark = True
    
    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    # prepare save path
    model_save_pth = f"{args.checkpoint}/{args.dataset}/{current_time}{create_save_path(args, mode = 'teacher')}"
    checkpoint_dir_name = model_save_pth

    if not os.path.isdir(model_save_pth):
        mkdir_p(model_save_pth)

    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(model_save_pth, "train.log")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Setting up logging folder : {model_save_pth}")

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes)
    model.cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)

    logging.info(f"Setting up optimizer : {args.optimizer}")

    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
    
    criterion = loss_dict[args.loss](loss=args.loss, beta=args.beta, gamma=args.gamma)
    test_criterion = torch.nn.CrossEntropyLoss()
    
    logging.info(f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)

    # set up logger
    logger = Logger(os.path.join(checkpoint_dir_name, "train_metrics.txt"))
    logger.set_names(["lr", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "SCE", "ECE"])

    start_epoch = args.start_epoch
    
    best_acc = 0.
    best_sce = float("inf")
    best_acc_stats = {"top1" : 0.0}
    best_cal_stats = {}

    for epoch in range(start_epoch, args.epochs):

        logging.info('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, get_lr(optimizer)))
        
        train_loss, top1_train = train(trainloader, model, optimizer, criterion)
        val_loss, top1_val, _, _, sce_score_val, ece_score_val = test(valloader, model, test_criterion)
        test_loss, top1, top3, top5, sce_score, ece_score = test(testloader, model, test_criterion)

        scheduler.step()

        logging.info("End of epoch {} stats: train_loss : {:.4f} | val_loss : {:.4f} | top1_train : {:.4f} | top1 : {:.4f} | SCE : {:.5f} | ECE : {:.5f}".format(
            epoch+1,
            train_loss,
            test_loss,
            top1_train,
            top1,
            sce_score,
            ece_score
        ))

        # save best accuracy model
        is_best = top1_val > best_acc
        best_acc = max(best_acc, top1_val)

        if sce_score < best_sce:
            best_sce = sce_score
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'dataset' : args.dataset,
                'model' : args.model
            }, False, checkpoint=model_save_pth, filename="best_calibration.pth")
            best_cal_stats = {
                "top1" : top1,
                "top3" : top3,
                "top5" : top5,
                "SCE" : sce_score,
                "ECE" : ece_score
            }

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'dataset' : args.dataset,
                'model' : args.model
            }, is_best, checkpoint=model_save_pth)
        
        # Update best stats
        if is_best:
            best_acc_stats = {
                "top1" : top1,
                "top3" : top3,
                "top5" : top5,
                "SCE" : sce_score,
                "ECE" : ece_score
            }
        
        logger.append([get_lr(optimizer), train_loss, top1_train, val_loss, top1_val, test_loss, top1, sce_score, ece_score])

    logging.info("training completed...")
    logging.info("The stats for best accuracy model on test set are as below:")
    logging.info(best_acc_stats)
    logging.info("The stats for best calibrated model on test set are as below:")
    logging.info(best_cal_stats)

    logger.append(["best_accuracy", 0, 0, 0, 0, 0, best_acc_stats["top1"], best_acc_stats["SCE"], best_acc_stats["ECE"]])
    logger.append(["best_calibration", 0, 0, 0, 0, 0, best_cal_stats["top1"], best_cal_stats["SCE"], best_cal_stats["ECE"]])

    # log results to a common file
    df = {
        "model" : [args.model],
        "dataset" : [args.dataset],
        "folder_path" : [checkpoint_dir_name],
        "acc_ECE" : [best_acc_stats["ECE"]],
        "acc_SCE" : [best_acc_stats["SCE"]],
        "acc_top1" : [best_acc_stats["top1"]],
        "cal_ECE" : [best_cal_stats["ECE"]],
        "cal_SCE" : [best_cal_stats["SCE"]],
        "cal_top1" : [best_cal_stats["top1"]],
        "checkpoint_train_loss" : [train_loss],
        "checkpoint_train_top1" : [top1_train],
        "checkpoint_val_loss" : [val_loss],
        "checkpoint_val_top1" : [top1_val],
        "checkpoint_test_loss" : [test_loss],
        "checkpoint_test_top1" : [top1],
        "checkpoint_test_top3" : [top3],
        "checkpoint_test_top5" : [top5],
        "checkpoint_test_sce" : [sce_score],
        "checkpoint_test_ece" : [ece_score]
    }
    df =  pd.DataFrame(df)
    save_path = os.path.join("results", "teacher_metrics.csv")
    df.to_csv(save_path, mode='a', index=False, header=(not os.path.exists(save_path)))
