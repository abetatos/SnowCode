# Imagedatagenerator
import pickle as pkl

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    
    def __init__(self, annotations_file, index_vars = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.index_vars = index_vars

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = self.img_labels.at[idx, "filename"]

        with open(img_path, "rb") as f:
            image = pkl.load(f)

        image, output = image[:-1], image[-1:]
        
        mask = np.where(output<0, 0., 1.)

        image = image if not self.index_vars else image[self.index_vars,]
        return image, mask, output
