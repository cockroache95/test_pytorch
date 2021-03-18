import os
from glob import glob
import random
from torch.utils.data import Dataset
import cv2
class FightDataset(Dataset):
    def __init__(self, root_dir, tranform=None):
        self.root_dir = root_dir
        self.data = []
        figs = glob(os.path.join(self.root_dir, "fighting/*"))
        for fig in figs:
            self.data.append({"image_path":fig, "label": 1})
        nofigs = glob(os.path.join(self.root_dir, "no_fight/*"))
        for nofig in nofigs:
            self.data.append({"image_path":nofig, "label": 0})

        self.transform = tranform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_tmp = self.data[idx]
        image = cv2.imread(sample_tmp["image_path"])
        label = sample_tmp["label"]
        sample = {"image": image, "label":label}
        if self.transform:
            sample = self.transform(sample)
        return sample