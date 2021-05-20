#!/usr/bin/env python
# coding: utf-8

# ---
# Model script

# In[4]:


import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import datasets
import torchvision
from tqdm.auto import tqdm
from torchvision.datasets import MNIST
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities.cli import LightningCLI
from argparse import ArgumentParser
import pathlib
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint


# In[5]:


class LitClassifier(pl.LightningModule):

    def __init__(self, learning_rate=0.001, training_dataset_folder = 'train', testing_dataset_folder = 'test', batch_size = 4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = torchvision.models.resnet18(pretrained=False)
        self.training_dataset_folder = training_dataset_folder
        self.testing_dataset_folder = testing_dataset_folder
        self.batch_size = batch_size
	 
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat,y)
        labels_hat = torch.argmax(y_hat, dim=1)
        training_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({'training_loss': loss,  'training_accuracy': training_acc}, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({'test_loss': loss, 'test_acc': test_acc}, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        optimizer2 = torch.optim.SGD(self.parameters(),lr=self.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1,step_size=10) # step size is random
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1,T_max=10) # Maximum number of iterations.
        scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1) # metric to be given
        return [optimizer1],[scheduler2]
        
    def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder(self.training_dataset_folder,transform=torchvision.transforms.Compose([torchvision.transforms.Resize((100,100)),torchvision.transforms.RandomRotation(5),torchvision.transforms.ToTensor()]))
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  shuffle=True,
                                                  batch_size=self.batch_size,
                                                  num_workers=4)
        return trainloader

    def test_dataloader(self):
        test_dataset = torchvision.datasets.ImageFolder(self.testing_dataset_folder,
                                                        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((100,100)),
             torchvision.transforms.ToTensor()]))
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=4)
        return testloader

