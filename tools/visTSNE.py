import pandas as pd
import os
from sklearn.manifold import TSNE
import time
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FeatDataset(Dataset):
    def __init__(self, data_root, slide_list):
        super(FeatDataset, self).__init__()

        self.data_root = data_root
        self.slide_list = slide_list

    def __len__(self):
        return  len(self.slide_list)

    def __getitem__(self, item):
        pt = torch.load(os.path.join(self.data_root, f"{self.slide_list[item]}.pt"))

        return pt


if __name__ == "__main__":
    lower_diagest = "/mnt/group-ai-medical-sz/private/jinxixiang/data/huayin1/patch_feature"
    upper_diagest = "/mnt/group-ai-medical-sz/private/jinxixiang/data/huayin2/patch_feature"
    diagest_csv = "/mnt/group-ai-medical-sz/private/zhongyiyang/data/huayin_digestive_tract_39229.csv"
    df = pd.read_csv(diagest_csv)
    all_diagest = df["slide_id"].values

    all_lower = sorted(os.listdir(lower_diagest))
    all_upper = sorted(os.listdir(upper_diagest))

    # remove ".pt"
    all_lower= [fname.split(".pt")[0] for fname in all_lower]
    all_upper = [fname.split(".pt")[0] for fname in all_upper]

    # csv and dir intersection
    all_upper = list(set(all_upper).intersection(set(all_diagest)))
    print(f"lower: {len(all_lower)} upper: {len(all_upper)}")

    lower_dataset = FeatDataset(lower_diagest, all_lower)
    lower_loader = DataLoader(lower_dataset, num_workers=8, pin_memory=True, batch_size=128)

    for idx, batch in enumerate(lower_loader):
        print(batch.shape)


    # fashion_tsne = TSNE(random_state=1112).fit_transform(x_subset)


