"""
CNN with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn

from models.image_classification.CNN_basic import CNN_basic
from models.registry import Model


@Model
class FoodIQ_CNN_basic(CNN_basic):
    """ See parent class for documentation"""

    def __init__(self, train_loader, **kwargs):
        """
        Creates a parent class model and replaces the classification layer with hydra heads for FoodIQ use-case.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The dataloader of the training set.
        """
        super().__init__(**kwargs)

        # Hydra heads: fully connected layers for classification (expected size: 288)
        self.hydra = nn.ModuleList()
        for k, v in train_loader.dataset.num_classes.items():
            # The postfix '_v7' is used to prevent attribute clashes with nn.Module (such as 'type')
            self.hydra.add_module(name=f'{k}_v7', module=nn.Linear(288, v))


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
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        # The :-3 is to remove the postfix added above in the hydra creation
        x = {k[:-3]:v(x) for k, v in self.hydra.named_children()}
        return x
