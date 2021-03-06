import os
import sys
sys.path.insert(0, "pytorch-image-models")
import timm
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from config import DefaultConfig
from utils.dataset.csv_dataset import CassavaDataset
from utils.dataset.cls_transforms import *


def train(config, fold_idx):
    model = CassvaImgClassifier(model_arch, config.n_classes, pretrained=True).to(device)

    # freeze parameters in bn layer
    for param in model.named_parameters():
        if ".bn" in param[0]:
            param[1].requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=1, eta_min=config.min_lr, last_epoch=-1)

    loss_tr = torch.nn.CrossEntropyLoss().to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    train_loader, valid_loader = get_dataloaders(config, fold_idx)
    for epoch in range(1, config.epochs+1):
        if epoch + 2 >= config.epochs:
            train_loader, _ = get_dataloaders(config, fold_idx, last=True)

        train_one_epoch(config, epoch, model, loss_tr, optimizer, train_loader, device,
                        scheduler=scheduler, schd_batch_update=False)

        with torch.no_grad():
            valid_one_epoch(config, epoch, model, loss_fn, valid_loader, device,
                            scheduler=None, schd_loss_update=False)

        torch.save(model.state_dict(), '{}_fold_{}_{}'.format(model_arch, fold_idx, epoch))

    del model, optimizer, train_loader, valid_loader, scheduler
    torch.cuda.empty_cache()


def train_one_epoch(config, epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)

        loss = loss_fn(image_preds, image_labels)
        loss.backward()

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) % config.accum_iter == 0) or ((step + 1) == len(train_loader)):
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


def get_cross_folds(config):
    data_csv = pd.read_csv(config.dataset_csv_path)

    if config.oversampling:
        img_num_in_class = [len(data_csv.loc[data_csv.label == i])
                            for i in range(5)]
        max_num = max(img_num_in_class)
        aug_ratio = [round(max_num / n) - 1 for n in img_num_in_class]

        for i, ratio in enumerate(aug_ratio):
            if ratio > 0:
                tmp = data_csv.loc[data_csv.label == i]
                data_csv = data_csv.append([tmp] * ratio)
        data_csv = data_csv.reset_index(drop=True)

    folds = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.SEED).split(
        np.arange(data_csv.shape[0]), data_csv.label.values)
    data_csv.loc[:, 'fold'] = 0

    return data_csv, folds


def get_dataloaders(config, fold_idx, last=False):
    data_pd, folds = get_cross_folds(DefaultConfig)
    for fold_number, (train_index, val_index) in enumerate(folds):
        data_pd.loc[data_pd.iloc[val_index].index, 'fold'] = fold_number

    if not last:
        train_dataset = CassavaDataset(config, pd=data_pd, image_ids=data_pd[data_pd['fold'] != fold_idx].index.values,
                                       mode="train", transforms=train_transforms(config))
    else:
        train_dataset = CassavaDataset(config, pd=data_pd, image_ids=data_pd[data_pd['fold'] != fold_idx].index.values,
                                       mode="train", transforms=valid_transforms(config))
    valid_dataset = CassavaDataset(config, pd=data_pd, image_ids=data_pd[data_pd['fold'] == fold_idx].index.values,
                                   mode="valid", transforms=valid_transforms(config))

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

    for i in range(5):
        print("="*20, i, "="*20)
        train(config, i)
