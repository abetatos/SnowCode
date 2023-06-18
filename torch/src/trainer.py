import os
import json
import time
from typing import List
from random import random
import shutil

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

import torch
from torch import device
from torch.utils.data import DataLoader

from src.losses import MSE, MAE
from src.datagen import CustomImageDataset
from src.models import NestNet, UpNet, MultiResUnet, AttentionUnet
from src.models import R2U_Net, RecU_Net, ResU_Net, DenseUnet, ResnetUnet
from harryplotter import HarryPlotter

class Trainer(HarryPlotter):

    @property
    def FEATURES(self):
        return json.loads(os.environ['Features'])

    def __init__(self, model, optimizer, annotations_path) -> None:
        assert model in {"UpNet", "NestNet", "MultiResUnet", "AttentionUnet",
                         "R2U_Net", "RecU_Net", "ResU_Net", "ResnetUnet"}
        assert optimizer in {"Adam"}
        self.model_name = model
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

        if optimizer == "Adam":
            self.opt = torch.optim.Adam

        self.annotations_df = annotations_path + "train_df.csv"
        self.annotations_df_test1 = annotations_path + "test_other_df.csv"
        self.annotations_df_test2 = annotations_path + "test_davos_df.csv"

        self.device = device('cuda') if torch.cuda.is_available() else device('cpu')
        self.epoch = 0
        self.val1 = None
        self.val2 = None
        self.loss = None

    def init_trainer(self, ini_channels=4, out_channels=1, width=4, depth=4, lr=1e-3, eps=1e-4):
        self.model = self.net(ini_channels, out_channels, width=width, depth=depth).to(self.device).double()
        self.optimizer = self.opt(self.model.parameters(), lr=lr, eps=eps)

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

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            labels, outputs = labels, outputs
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
    
    def _generate_report(self, globloss, globval1, globval2, t0, index_vars):
        mean_array = np.mean([[l, v1, v2] for l, v1, v2 in zip(globloss, globval1, globval2)], axis=1)
        i = np.argmin(mean_array)
        best_val = globval2[i]

        features = [self.FEATURES[iv] for iv in index_vars]
        report = [i, globloss[i], globval1[i], best_val, time.time() - t0, self.model_name, features]
        df = pd.DataFrame([report], columns=['Epoch', 'Train loss', "Val loss 1", "Val loss 2", "Train time", "Model", "Features"])
        df.to_csv(f"Models/{round(t0)}_best_report.csv")

        report = [[loss, val1, val2] for loss, val1, val2 in zip(globloss, globval1, globval2)]
        df = pd.DataFrame(report, columns=['Train loss', "Val loss 1", "Val loss 2"])
        df.to_csv(f"Models/{round(t0)}_train_report.csv")

        val100 = round(best_val*100)
        ivars = ''.join([str(i) for i in index_vars])
        os.rename(f"tmp/model_{i}.pt", f"Models/{round(t0)}_model_{val100}_{ivars}.pt")
        shutil.rmtree("tmp")

    def train(self, batch_size, n_epochs, index_vars=None):
        self.plot_bool = [i in index_vars if index_vars else True for i in range(self.N_FEATURES)]
        return self._train(batch_size, n_epochs, index_vars)
    
    def clear_model(self): 
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()

    def _train(self, batch_size, n_epochs, index_vars=None):
        t0 = time.time()
        cid = CustomImageDataset(self.annotations_df, index_vars)
        trainloader = DataLoader(cid, batch_size=batch_size, shuffle=False)
        validationloader1 = DataLoader(CustomImageDataset(self.annotations_df_test1, index_vars), batch_size=batch_size, shuffle=False)
        validationloader2 = DataLoader(CustomImageDataset(self.annotations_df_test2, index_vars), batch_size=batch_size, shuffle=False)

        glob_val1, glob_val2, glob_loss = [], [], []
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            
            loss = self.train_epoch(trainloader)
            glob_loss.append(loss)

            train_loss = self.validate_epoch(trainloader)

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

        self._generate_report(glob_loss, glob_val1, glob_val2, t0, index_vars)

        self.plot_evolution(glob_loss, glob_val1, title="Validation 1")
        self.plot_evolution(glob_loss, glob_val2, title="Validation 2")
        return glob_val1, glob_val2, glob_loss

    def load_checkpoint(self, path): 
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])