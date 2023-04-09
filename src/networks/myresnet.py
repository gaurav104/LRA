import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torch
import sys

class MyResNet34(nn.Module):
    def __init__(self, num_classes=200, in_channels=3):
        super(MyResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        # for fine-tuning freeze early layers
        '''
        for count, child in enumerate(self.resnet.children()):
            if count < 7:
                for param in child.parameters():
                    param.requires_grad = False
        '''
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Linear(512, num_classes)
        self.resnet.conv1 = nn.Conv2d(in_channels,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.resnet.maxpool = nn.Sequential()

    def forward(self, x, out_feature=False):
        # change forward here
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        feature2 = x
        x = self.resnet.layer2(x)
        feature3 = x
        x = self.resnet.layer3(x)
        feature4 = x
        x = self.resnet.layer4(x)
        feature5 = x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        feature = x
        x = self.resnet.fc(x)
               
        if out_feature == False:
            return x
        else:
            return x,[feature2, feature3, feature4, feature5, feature]

class MyResNet18(nn.Module):
    def __init__(self, num_classes=200, in_channels=3):
        super(MyResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        # for fine-tuning freeze early layers
        '''
        for count, child in enumerate(self.resnet.children()):
            if count < 7:
                for param in child.parameters():
                    param.requires_grad = False
        '''
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Linear(512, num_classes)
        self.resnet.conv1 = nn.Conv2d(in_channels,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.resnet.maxpool = nn.Sequential()

    def forward(self, x, out_feature=False):
        # change forward here
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        feature2 = x
        x = self.resnet.layer2(x)
        feature3 = x
        x = self.resnet.layer3(x)
        feature4 = x
        x = self.resnet.layer4(x)
        feature5 = x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        feature = x
        x = self.resnet.fc(x)
               
        if out_feature == False:
            return x
        else:
            return x,[feature2, feature3, feature4, feature5, feature]