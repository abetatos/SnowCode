import os
import json
import time
from typing import List
from random import random
import shutil

import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np

import torch
from torch import device
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, Adagrad

from .losses import MSE, MAE
from .datagen import CustomImageDataset
from .models import *
from harryplotter import HarryPlotter


class Trainer(HarryPlotter):

    @property
    def FEATURES(self):
        return json.loads(os.environ['Features'])

    def __init__(self, model, optimizer, annotations_path) -> None:
        assert model in {"UpNet", "NestNet", "MultiResUnet", "AttentionUnet",
                         "R2U_Net", "RecU_Net", "ResU_Net", "ResnetUnet"}
        assert optimizer in {"Adam", "SGD", "Adagrad"}

        self.model_name = model
        self.optimizer_name = optimizer
        self.net = {
            "UpNet": UpNet,
            "NestNet": NestNet,
            "MultiResUnet": MultiResUnet,
            "AttentionUnet": AttentionUnet,
            "R2U_Net": R2U_Net, 
            "RecU_Net": RecU_Net, 
            "ResU_Net": ResU_Net,
            "DenseUnet": DenseUnet,
            "ResnetUnet": ResnetUnet
            }[model]

        self.opt = {
            "Adam": Adam, 
            "SGD": SGD,
            "Adagrad": Adagrad
        }[optimizer]

        self.annotations_df_pretr =  annotations_path + "pretrain_df.csv"
        self.annotations_df = annotations_path + "train_df.csv"
        self.annotations_df_test1 = annotations_path + "test_other_df.csv"
        self.annotations_df_test2 = annotations_path + "test_davos_df.csv"

        self.device = device('cuda') if torch.cuda.is_available() else device('cpu')
        self.epoch = 0
        self.val1 = None
        self.val2 = None
        self.loss = None

    def _instantiate_optimizer(self, lr, eps, momentum, weight_decay, lr_decay, nesterov):
        if self.optimizer_name == "Adam":
            self.optimizer = self.opt(self.model.parameters(), lr=lr, eps=eps)
        elif self.optimizer_name == "SGD":
            self.optimizer = self.opt(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        elif self.optimizer_name == "Adagrad":
            self.optimizer = self.opt(self.model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)

    def update_optimizer(self, optimizer,  lr=1e-3, eps=1e-8, momentum=0.9, weight_decay=1e-5, lr_decay=1e-3, nesterov=True):
        self.opt = {
            "Adam": Adam, 
            "SGD": SGD,
            "Adagrad": Adagrad
        }[optimizer]
        self.optimizer_name = optimizer
        self._instantiate_optimizer(lr, eps, momentum, weight_decay, lr_decay, nesterov)
        
    def init_trainer(self, ini_channels=4, out_channels=1, width=4, depth=4, lr=1e-3, eps=1e-8, momentum=0.9, weight_decay=1e-5, lr_decay=1e-3, nesterov=True):
        self.model = self.net(ini_channels, out_channels, width=width, depth=depth).to(self.device).double()
        self._instantiate_optimizer(lr, eps, momentum, weight_decay, lr_decay, nesterov)

    def save_model(self, loss, val1, val2, path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'val1': [val1, val2]
        }, path)

    def train_epoch(self, trainloader):
        n_items = len(trainloader) 
        running_loss = []
        for i, (inputs, mask, labels) in tqdm(enumerate(trainloader, 0), total=n_items):
            # get the inputs; data is a list of [inputs, labels]
            inputs, mask, labels = inputs.to(self.device), mask.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad() # zero the parameter gradients
            
            outputs = self.model(inputs)
            loss = MSE(outputs, labels.double(), mask).double()

            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if random() < float(os.environ.get('displayprob', 1e-3)):
                self.plot_inputs(inputs)
                self.plot_outputs(labels, outputs)
                self.plot_heatmap(labels, outputs, mask)

        return np.array(running_loss).mean()

    def validate_epoch(self, validationloader): 
        running_validation = []
        for i, (inputs, mask, labels) in tqdm(enumerate(validationloader, 0), total=len(validationloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, mask, labels = inputs.to(self.device), mask.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                valm = MAE(outputs, labels, mask)
                running_validation.append(valm.cpu())
        return np.array(running_validation).mean()
    
    def _generate_report(self, globloss, train_loss, globval1, globval2, t0, index_vars):
        mean_array = np.mean([[v1, v2] for v1, v2 in zip(globval1, globval2)], axis=1)
        i = np.argmin(mean_array)
        best_val = globval2[i]

        features = [self.FEATURES[iv] for iv in index_vars]
        report = [i, globloss[i], train_loss[i], globval1[i], best_val, time.time() - t0, self.model_name, features]
        df = pd.DataFrame([report], columns=['Epoch', 'Train loss', "Train val", "Val loss 1", "Val loss 2", "Train time", "Model", "Features"])
        df.to_csv(f"Models/{round(t0)}_best_report.csv")

        report = [[loss, tl, val1, val2] for loss, tl, val1, val2 in zip(globloss, train_loss, globval1, globval2)]
        df = pd.DataFrame(report, columns=['Train loss', "Train val", "Val loss 1", "Val loss 2"])
        df.to_csv(f"Models/{round(t0)}_train_report.csv")

        val100 = round(best_val*100)
        ivars = ''.join([str(i) for i in index_vars])
        modelname = f"Models/{round(t0)}_model_{val100}_{ivars}.pt"
        os.rename(f"tmp/model_{i+1}.pt", modelname)
        shutil.rmtree("tmp")
        return modelname
    
    def train(self, batch_size, n_epochs, index_vars=None):
        self.plot_bool = [i in index_vars if index_vars else True for i in range(self.N_FEATURES)]
        return self._train(batch_size, n_epochs, index_vars)
    
    def clear_model(self): 
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()

    def pretrain(self, batch_size, n_epochs, index_vars = None):
        print("Pretrain")
        self.plot_bool = [i in index_vars if index_vars else True for i in range(self.N_FEATURES)]
        pretrain_loader =  DataLoader(CustomImageDataset(self.annotations_df_pretr, index_vars), batch_size=batch_size, shuffle=False)
        for epoch in range(n_epochs): 
            self.train_epoch(pretrain_loader)

    def _train(self, batch_size, n_epochs, index_vars=None):
        t0 = time.time()
        trainloader = DataLoader(CustomImageDataset(self.annotations_df, index_vars), batch_size=batch_size, shuffle=False)
        validationloader1 = DataLoader(CustomImageDataset(self.annotations_df_test1, index_vars), batch_size=batch_size, shuffle=False)
        validationloader2 = DataLoader(CustomImageDataset(self.annotations_df_test2, index_vars), batch_size=batch_size, shuffle=False)

       
        glob_val1, glob_val2, glob_loss, glob_train_loss = [], [], [], []
        for epoch in range(1, n_epochs+1):  # loop over the dataset multiple times
            
            loss = self.train_epoch(trainloader)
            glob_loss.append(loss)

            train_loss = self.validate_epoch(trainloader)
            glob_train_loss.append(train_loss)

            val_loss1 = self.validate_epoch(validationloader1)
            glob_val1.append(val_loss1)

            val_loss2 = self.validate_epoch(validationloader2)
            glob_val2.append(val_loss2)

            self.epoch = epoch
            os.makedirs("tmp", exist_ok=True)
            self.save_model(loss, val_loss1, val_loss2, path = f"tmp/model_{epoch}.pt")

            print(f"╔══════════════╦════════════╦═══════════════╦═══════════════╗")
            print(f"║  Epoch       ║  TrainVal  ║  Validation2  ║  Validation1  ║")
            print(f"╠══════════════╬════════════╬═══════════════╬═══════════════╣")
            print(f"║  {epoch:<12}║  {train_loss:<10.3f}║  {val_loss2:<13.3f}║  {val_loss1:<13.3f}║")
            print(f"╚══════════════╩════════════╩═══════════════╩═══════════════╝")

        modelname = self._generate_report(glob_loss, glob_train_loss, glob_val1, glob_val2, t0, index_vars)

        self.plot_evolution(glob_loss, glob_val1, title="Validation 1")
        self.plot_evolution(glob_loss, glob_val2, title="Validation 2")
        return glob_val1, glob_val2, glob_loss, modelname

    def load_checkpoint(self, path): 
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])