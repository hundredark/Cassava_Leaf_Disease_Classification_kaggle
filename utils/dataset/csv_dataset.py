import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.loss import label_smooth


def rand_bbox(size, lam):
    W, H = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self, config, pd, image_ids, mode="train", transforms=None):
        super().__init__()
        self.config     = config
        self.pd         = pd
        self.image_ids  = image_ids
        self.mode       = mode
        self.transforms = transforms

        # self.data_root = data_root
        # self.do_fmix = do_fmix
        # self.fmix_params = fmix_params
        # self.do_cutmix = do_cutmix
        # self.cutmix_params = cutmix_params
        #
        # self.output_label = output_label
        # self.one_hot_label = one_hot_label
        #
        # if output_label == True:
        #     self.labels = self.df['label'].values
        #     # print(self.labels)
        #
        #     if one_hot_label is True:
        #         self.labels = np.eye(self.df['label'].max() + 1)[self.labels]
        #         # print(self.labels)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_name, label, _, _, _ = self.pd.iloc[image_id]
        img_path = os.path.join(self.config.dataset_img_dir, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.uint8)

        # pos, neg = label_smooth(self.config.label_smooth_eps)
        # label_vector = torch.zeros((1, self.config.n_classes)) + neg
        # label_vector[int(label)] = pos

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, label  # _vector
