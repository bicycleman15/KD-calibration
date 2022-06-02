from .resnet import resnet18, resnet34, resnet50, resnet110, resnet152
from .convnet import ConvNet
# from .resnet import resnet20, resnet56
# from .resnet_tinyimagenet import resnet34, resnet50

model_dict = {
    # resnet models can be used for cifar10/100, svhn
    "resnet18" : resnet18,
    "resnet34" : resnet34,
    "resnet50" : resnet50,
    "resnet110" : resnet110,
    "resnet152" : resnet152,
    "convnet" : ConvNet
    # "resnet20" : resnet20,
    # "resnet56" : resnet56,

    # resnet_tinyimagenet models
    # "resnet34_imagenet" : resnet34,
    # "resnet50_imagenet" : resnet50
}