import os
import sys
sys.path.insert(0, "pytorch-image-models")
import timm
import time
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler

from config import DefaultConfig
from utils.dataset.csv_dataset import CassavaDataset
from utils.dataset.cls_transforms import *

#
# from glob import glob
# from sklearn.model_selection import GroupKFold, StratifiedKFold
# from skimage import io
#
# from datetime import datetime
# import torchvision
# from torchvision import transforms
#
#
#
# import matplotlib.pyplot as plt
# from torch.utils.data.sampler import SequentialSampler, RandomSampler
# from torch.nn.modules.loss import _WeightedLoss
# import torch.nn.functional as F
#
#
# import sklearn
# import warnings
# import joblib
# from sklearn.metrics import roc_auc_score, log_loss
# from sklearn import metrics
# import warnings
# import cv2
# import pydicom
# #from efficientnet_pytorch import EfficientNet
# from scipy.ndimage.interpolation import zoom
#
# from FMix.fmix import sample_mask, make_low_freq_image, binarise_mask


def train(config):
    model = CassvaImgClassifier(model_arch, config.n_classes, pretrained=True).to(device)

    # scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=1, eta_min=config.min_lr, last_epoch=-1)

    loss_tr = torch.nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(1, config.epochs+1):
        if epoch + 2 > config.epochs:
            train_loader, _ = get_dataloaders(config, last=True)
        else:
            train_loader, valid_loader = get_dataloaders(config)

        train_one_epoch(config, epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler,
                        schd_batch_update=False)

        with torch.no_grad():
            valid_one_epoch(config, epoch, model, loss_fn, valid_loader, device, scheduler=None, schd_loss_update=False)

        torch.save(model.state_dict(), '{}_fold_{}_{}'.format(model_arch, config.fold_idx, epoch))

    del model, optimizer, train_loader, valid_loader, scheduler
    torch.cuda.empty_cache()

def train_one_epoch(config, epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        # print(image_labels.shape, exam_label.shape)
        # with autocast():
        image_preds = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)

        loss = loss_fn(image_preds, image_labels)
        loss.backward()
        #    scaler.scale(loss).backward()

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) % config.accum_iter == 0) or ((step + 1) == len(train_loader)):
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None and schd_batch_update:
                scheduler.step()

        if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {running_loss:.4f}'

            pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(config, epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum / sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()


def get_dataloaders(config, last=False):
    if not last:
        train_dataset = CassavaDataset(config, mode="train", transforms=train_transforms(config))
    else:
        train_dataset = CassavaDataset(config, mode="train", transforms=valid_transforms(config))
    valid_dataset = CassavaDataset(config, mode="valid", transforms=valid_transforms(config))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.num_workers)
    return train_loader, valid_loader


class CassvaImgClassifier(torch.nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


def fix_seed(config):
    os.environ['PYTHONHASHSEED'] = str(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == "__main__":
    model_arch = "tf_efficientnet_b4_ns"
    device= torch.device('cuda:0')
    config = DefaultConfig
    fix_seed(config)
    
    '''
    json_path = "../dataset/label_num_to_disease_map.json"
    with open(json_path, 'r') as load_f:
        label2int = dict(json.load(load_f))
    print(label2int)
    '''

    train(config)
