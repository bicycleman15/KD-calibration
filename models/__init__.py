from .resnet import resnet18, resnet34, resnet50, resnet110, resnet152
from .convnet import ConvNet
# from .resnet import resnet20, resnet56
from .resnet_tinyimagenet import resnet50 as resnet50_tin
from .resnet_tinyimagenet import resnet34 as resnet34_tin
from .resnet_tinyimagenet import resnet18 as resnet18_tin

model_dict = {
    # resnet models can be used for cifar10/100 32x32 images
    "resnet18" : resnet18,
    "resnet34" : resnet34,
    "resnet50" : resnet50,
    "resnet110" : resnet110,
    "resnet152" : resnet152,
    "convnet" : ConvNet,

    # resnet_tinyimagenet models 64x64 images
    "resnet34_tin" : resnet34_tin,
    "resnet50_tin" : resnet50_tin,
    "resnet18_tin" : resnet18_tin,
}