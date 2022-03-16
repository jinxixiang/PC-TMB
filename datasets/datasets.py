import os
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from .BatchWSI import BatchWSI


class GraphDataset(Dataset):
    def __init__(self, df, labels, cfg):
        super(GraphDataset, self).__init__()
        self.df = df
        self.labels = labels
        self.cfg = cfg
        self.feat_dir = cfg.Data.dataset.feat_dir
        self.type_dict = cfg.Data.dataset.type_dict
        self.num_type = len(self.type_dict.keys())

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return self.labels

    def __getitem__(self, item):
        file_name = self.df['image_id'].values[item]
        pt_dir = os.path.join(self.feat_dir, f"{file_name}.pt")
        feat = torch.load(pt_dir)
        bag = BatchWSI.from_data_list([feat])

        type = self.type_dict[self.df["type"].values[item]]
        type_tensor = F.one_hot(torch.tensor([type]),
                                num_classes=self.num_type).squeeze()

        label = torch.tensor(self.labels[item]).float()

        return bag, type_tensor.float(), label.long()


