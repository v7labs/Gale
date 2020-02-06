"""
Model definition adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import logging

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.image_classification.ResNet import ResNet, _BasicBlock, model_urls, _Bottleneck
from models.registry import Model


class FoodIQ_ResNet(ResNet):

    expected_input_size = (224, 224)

    def __init__(self, block, train_loader=None, num_classes=None, **kwargs):
        """
        Creates a parent class model and replaces the classification layer with hydra heads for FoodIQ use-case.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The dataloader of the training set.
        """
        super().__init__(block=block, num_classes=1, **kwargs)  # output_channels=1 is deliberate as its not used!

        # Hydra heads: fully connected layers for classification (expected size: 512 * block.expansion)
        self.hydra = nn.ModuleList()
        if train_loader is not None:
            # This happens at train time, where the train loader exists
            d = train_loader.dataset.num_classes
        elif type(num_classes) == dict:
            # This is used at inference time, where the number of classes is known only trough loading the model.pth
            d = {k:len(v) for k, v in num_classes.items()}
        else:
            raise AttributeError
        for k, v in d.items():
            # The postfix '_v7' is used to prevent attribute clashes with nn.Module (such as 'type')
            self.hydra.add_module(name=f'{k}_v7', module=nn.Linear(512 * block.expansion, v))

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        dict
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.ablate:
            return x
        else:
            # The :-3 is to remove the postfix added above in the hydra creation
            x = {k[:-3]: v(x) for k, v in self.hydra.named_children()}
            return x


@Model
def FoodIQ_resnet18(pretrained=False, **kwargs):
    """Constructs a _ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FoodIQ_ResNet(block=_BasicBlock,  layers=[2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model
FoodIQ_resnet18.expected_input_size = FoodIQ_ResNet.expected_input_size


@Model
def FoodIQ_resnet34(pretrained=False, **kwargs):
    """Constructs a _ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FoodIQ_ResNet(block=_BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model
FoodIQ_resnet34.expected_input_size = FoodIQ_ResNet.expected_input_size


@Model
def FoodIQ_resnet50(pretrained=False, **kwargs):
    """Constructs a _ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FoodIQ_ResNet(block=_Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model
FoodIQ_resnet50.expected_input_size = FoodIQ_ResNet.expected_input_size


@Model
def FoodIQ_resnet101(pretrained=False, **kwargs):
    """Constructs a _ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FoodIQ_ResNet(block=_Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model
FoodIQ_resnet101.expected_input_size = FoodIQ_ResNet.expected_input_size


@Model
def FoodIQ_resnet152(pretrained=False, **kwargs):
    """Constructs a _ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FoodIQ_ResNet(block=_Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model
FoodIQ_resnet152.expected_input_size = FoodIQ_ResNet.expected_input_size