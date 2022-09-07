#!/usr/bin/env python
# coding: utf-8
# %%


from dataset.dataset import SequenceDataset_per
from distortions import (block_wise, color_contrast, color_saturation,
                         gaussian_blur, gaussian_noise_color, jpeg_compression,
                         video_compression)
from pretrainedmodels import xception
from RFM.utils1.utils import data_prefetcher_two, cal_fam, setup_seed, calRes
from Video_Transformer.models.videotransformer import VideoTransformer
from Video_Transformer.models.imagetransformer import ImageTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
# from catalyst.data import BalanceClassSampler
from torch.autograd import Variable
import time
from tensorboardX import SummaryWriter
import glob
from datetime import datetime
import socket
import matplotlib.pyplot as plt
from torchsampler import ImbalancedDatasetSampler
from xception import xception

import glob
from itertools import chain
import os
import cv2
import random
import zipfile
import os.path as osp
import pandas as pd

from functools import reduce
from einops import rearrange, repeat
import torch.optim as optim
import torchvision
# from linformer import Linformer
from PIL import Image
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

# %%
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from utils.utils import Logger, AverageMeter, calculate_metrics, Test_time_agumentation
from network.models import model_selection
from dataset.dataset import AllDataset, CelebDFDataset, SequenceDataset, SequenceDataset_CelebDF, SequenceDataset_DFDC, SequenceDataset_DFDC_new, SequenceDataset_phase
from loss.losses import LabelSmoothing, SingleCenterLoss
from efficientnet_pytorch_3d import EfficientNet3D
from time_transformer import TimeTransformer
from effnetv2 import effnetv2_s, effnetv2_m, effnetv2_l, effnetv2_xl
from config import config as my_cfg
from model_core import Two_Stream_Net
from loss1.am_softmax import AMSoftmaxLoss
from networks.xception import TransferModel
from MAT import MAT
from f3net import F3Net
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, CenterCrop,\
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise,\
    GaussianBlur, Resize, Normalize, RandomRotate90, Cutout, GridDropout, CoarseDropout, MedianBlur


# %%
def train_model(model, criterion, optimizer, epoch):
    model.cuda()
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    training_process = tqdm(train_loader)
     for i, (XI, label) in enumerate(training_process):
          torch.cuda.empty_cache()
           if i > 0:
                training_process.set_description("Epoch: %d, Loss: %.4f, Acc: %.4f" % (
                    epoch, losses.avg.item(), accuracies.avg.item()))

            x = Variable(XI.cuda(device_id))
            label = Variable(label.cuda(device_id))
            # label = Variable(torch.LongTensor(label).cuda(device_id))
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)
            y_pred = nn.Softmax(dim=1)(y_pred)

            # Compute and print loss
            loss = criterion(y_pred, label)
     #         loss = ce_scl(y_pred, label)
            acc = calculate_metrics(nn.Softmax(
                dim=1)(y_pred).cpu(), label.cpu())
            losses.update(loss.cpu(), x.size(0))
            accuracies.update(acc, x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    scheduler.step()
#     train_logger.log(phase="train", values={
#         'epoch': epoch,
#         'loss': format(losses.avg.item(), '.4f'),
#         'acc': format(accuracies.avg.item(), '.4f'),
#         'lr': optimizer.param_groups[0]['lr']
#     })
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}". format(
        losses.avg, accuracies.avg))
    writer.add_scalars('loss', {'log/train_loss_epoch': losses.avg}, epoch)
    writer.add_scalars('acc', {'log/train_acc_epoch': accuracies.avg}, epoch)
    return accuracies.avg.item(), losses.avg.item()


# %%
# 9 times
def TTA(model_, img, activation=nn.Softmax(dim=1)):
    # original 1
    outputs = activation(model_(img))
    tta = Test_time_agumentation()
    # 水平翻转 + 垂直翻转 2
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        outputs += activation(model_(flip_img))
    # 2*3=6
    for flip_img in [img, flip_imgs[0]]:
        rot_flip_imgs = tta.tensor_rotation(flip_img)
        for rot_flip_img in rot_flip_imgs:
            outputs += activation(model_(rot_flip_img))

    outputs /= 9

    return outputs


def eval_model(model, epoch, eval_loader, is_save=True, is_tta=False, metric_name='acc'):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    eval_process = tqdm(eval_loader)
    labels = []
    outputs = []
    val_criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            if i > 0:
                eval_process.set_description("Epoch: %d, Loss: %.4f, Acc: %.4f" %
                                             (epoch, losses.avg.item(), accuracies.avg.item()))
            img, label = Variable(img.cuda(device_id)), Variable(
                label.cuda(device_id))
            if not is_tta:
                y_pred = model(img)
                y_pred = nn.Softmax(dim=1)(y_pred)
            else:
                y_pred = TTA(model, img, activation=nn.Softmax(dim=1))
            outputs.append(1-y_pred[:, 0])  # 扣掉真的機率 留下假的機率
            labels.append(label)
#             loss = ce_scl(y_pred, label)
            loss = val_criterion(y_pred, label)
            acc = calculate_metrics(
                y_pred.cpu(), label.cpu(), metric_name=metric_name)
            losses.update(loss.cpu(), img.size(0))
            accuracies.update(acc, img.size(0))
    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    labels[labels > 0] = 1
    AUC = roc_auc_score(labels, outputs)

#     if is_save:
#         train_logger.log(phase="val", values={
#             'epoch': epoch,
#             'loss': format(losses.avg.item(), '.4f'),
#             'acc': format(accuracies.avg.item(), '.4f'),
#             'lr': optimizer.param_groups[0]['lr']
#         })

    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(
        losses.avg, accuracies.avg))
    print('AUC:', AUC)
    writer.add_scalars('loss', {'log/val_loss_epoch': losses.avg}, epoch)
    writer.add_scalars('acc', {'log/val_acc_epoch': accuracies.avg}, epoch)
    return AUC, accuracies.avg.item(), losses.avg.item()  # accuracies.avg


def Inference(model, eval_loader, is_tta=False, metric_name='acc'):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    eval_process = tqdm(eval_loader)
    labels = []
    outputs = []
    val_criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            img, label = Variable(img.cuda(device_id)), Variable(
                label).cuda(device_id)
            if not is_tta:
                y_pred = model(img)
                y_pred = nn.Softmax(dim=1)(y_pred)
            else:
                y_pred = TTA(model, img, activation=nn.Softmax(dim=1))
            outputs.append(1-y_pred[:, 0])  # 扣掉真的機率 留下假的機率
            labels.append(label)
            loss = val_criterion(y_pred, label)
            acc = calculate_metrics(
                y_pred.cpu(), label.cpu(), metric_name=metric_name)
            losses.update(loss.cpu(), img.size(0))
            accuracies.update(acc, img.size(0))
    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    labels[labels > 0] = 1
    AUC = roc_auc_score(labels, outputs)
    print('Inference AUC:', AUC)
    print('Inference loss:{0},  Inference acc:{1}'.format(
        losses.avg, accuracies.avg))
    return AUC


# %%
def save_loss(train_loss_history, val_loss_history, save_dir):
    # plotting loss
    print("Saving loss history ...")
    plt.figure()
    plt.plot(train_loss_history, label="Training loss")
    plt.plot(val_loss_history, label="Validation loss")
    plt.xlabel('Iteration')
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_history.png"))
    np.save(os.path.join(save_dir, "Train_loss_history"), train_loss_history)
    np.save(os.path.join(save_dir, "Val_loss_history"), val_loss_history)


def save_acc(train_acc_record, valid_acc_record, save_dir):
    # plotting loss
    print("Saving acc history ...")
    plt.figure()
    plt.plot(train_acc_record, label="Training acc")
    plt.plot(valid_acc_record, label="Validation acc")
    plt.xlabel('Iteration')
    plt.ylabel("acc")
    plt.title("acc history")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "acc_history.png"))
    np.save(os.path.join(save_dir, "Train_acc_history"), train_acc_record)
    np.save(os.path.join(save_dir, "Val_acc_history"), valid_acc_record)


# %%

seed = 23


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

# %%
celebdf_real = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/Celeb-DF-v2/Celeb-real'
celebdf_real1 = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/Celeb-DF-v2/YouTube-real'
celebdf_fake = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/Celeb-DF-v2/Celeb-synthesis'

train_celebd_real = glob.glob(os.path.join(celebdf_real, '*', '*'))
train_celebd_real1 = glob.glob(os.path.join(celebdf_real1, '*', '*'))
train_celebdf_fake = glob.glob(os.path.join(celebdf_fake, '*', '*'))
np.random.shuffle(train_celebd_real)
np.random.shuffle(train_celebd_real1)
np.random.shuffle(train_celebdf_fake)
celebdf_list = []
celebdf_list.extend(train_celebd_real[:200])
celebdf_list.extend(train_celebd_real1[:200])
celebdf_list.extend(train_celebdf_fake[:600])
print(f"Train celebdf_list: {len(celebdf_list)}")
np.random.shuffle(celebdf_list)

dfdc = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/dfdc'
train_dfdc = glob.glob(os.path.join(dfdc, '*', '*'))
np.random.shuffle(train_dfdc)
dfdc_list = []
dfdc_list.extend(train_dfdc[:1000])
print(f"Train dfdc_list: {len(dfdc_list)}")
np.random.shuffle(dfdc_list)

# %%
train_dir_real = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/RealFF'
train_dir_fake = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/FakeFF/FaceSwap/'
train_dir_fake_2 = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/FakeFF/Face2Face/'
train_dir_fake_3 = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/FakeFF/Deepfakes/'
train_dir_fake_4 = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/FakeFF/NeuralTextures/'

train_list_real = glob.glob(os.path.join(train_dir_real, '*', '*'))
train_list_fake = glob.glob(os.path.join(train_dir_fake, '*', '*'))
train_list_fake_2 = glob.glob(os.path.join(train_dir_fake_2, '*', '*'))
train_list_fake_3 = glob.glob(os.path.join(train_dir_fake_3, '*', '*'))
train_list_fake_4 = glob.glob(os.path.join(train_dir_fake_4, '*', '*'))

train_list_real, valid_list_real = train_test_split(
    train_list_real, random_state=seed, train_size=0.6)
train_list_fake, valid_list_fake = train_test_split(
    train_list_fake, random_state=seed, train_size=0.6)
train_list_fake_2, valid_list_fake_2 = train_test_split(
    train_list_fake_2, random_state=seed, train_size=0.6)
train_list_fake_3, valid_list_fake_3 = train_test_split(
    train_list_fake_3, random_state=seed, train_size=0.6)
train_list_fake_4, valid_list_fake_4 = train_test_split(
    train_list_fake_4, random_state=seed, train_size=0.6)

np.random.shuffle(train_list_real)
np.random.shuffle(train_list_fake)
np.random.shuffle(train_list_fake_2)
np.random.shuffle(train_list_fake_3)
np.random.shuffle(train_list_fake_4)
np.random.shuffle(train_list_real)
np.random.shuffle(train_list_fake)
np.random.shuffle(train_list_fake_2)
np.random.shuffle(train_list_fake_3)
np.random.shuffle(train_list_fake_4)


np.random.shuffle(valid_list_real)
np.random.shuffle(valid_list_fake)
np.random.shuffle(valid_list_fake_2)
np.random.shuffle(valid_list_fake_3)
np.random.shuffle(valid_list_fake_4)
np.random.shuffle(valid_list_real)
np.random.shuffle(valid_list_fake)
np.random.shuffle(valid_list_fake_2)
np.random.shuffle(valid_list_fake_3)
np.random.shuffle(valid_list_fake_4)

train_list = []
train_list.extend(train_list_real[:100000])
train_list.extend(train_list_fake[:25000])
train_list.extend(train_list_fake_2[:25000])
train_list.extend(train_list_fake_3[:25000])
train_list.extend(train_list_fake_4[:25000])

valid_list = []
valid_list.extend(valid_list_real[:20000])
valid_list.extend(valid_list_fake[:5000])
valid_list.extend(valid_list_fake_2[:5000])
valid_list.extend(valid_list_fake_3[:5000])
valid_list.extend(valid_list_fake_4[:5000])

print(f"Train Data Real: {len(train_list_real)}")
print(f"Train Data Fake: {len(train_list_fake)}")
print(f"Train Data Fake 2: {len(train_list_fake_2)}")
print(f"Train Data Fake 3: {len(train_list_fake_3)}")
print(f"Train Data Fake 4: {len(train_list_fake_4)}")

np.random.shuffle(train_list)
np.random.shuffle(train_list)
np.random.shuffle(train_list)
np.random.shuffle(train_list)

np.random.shuffle(valid_list)
np.random.shuffle(valid_list)
np.random.shuffle(valid_list)
np.random.shuffle(valid_list)


def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        HorizontalFlip(),
        GaussNoise(p=0.1),
        GaussianBlur(p=0.1),
        RandomRotate90(),
        Resize(height=size, width=size),
        PadIfNeeded(min_height=size, min_width=size,
                    border_mode=cv2.BORDER_CONSTANT),
        RandomCrop(height=size, width=size),
        OneOf([RandomBrightnessContrast(), FancyPCA(),
              HueSaturationValue()], p=0.5),
        OneOf([CoarseDropout(), GridDropout()], p=0.5),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                         rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )


def create_val_transforms(size=300):
    return Compose([
        Resize(height=size, width=size),
        # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        # GaussianBlur(blur_limit=3, p=1),
        # CenterCrop(height=size, width=size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# %%
def get_distortion_function(dist_type):
    func_dict = dict()  # a dict of function
    func_dict['CS'] = color_saturation
    func_dict['CC'] = color_contrast
    func_dict['BW'] = block_wise
    func_dict['GNC'] = gaussian_noise_color
    func_dict['GB'] = gaussian_blur
    func_dict['JPEG'] = jpeg_compression
    return func_dict[dist_type]


def get_distortion_parameter(dist_type, level):
    param_dict = dict()  # a dict of list
    param_dict['CS'] = [0.4, 0.3, 0.2, 0.1, 0.0]  # smaller, worse
    param_dict['CC'] = [0.85, 0.725, 0.6, 0.475, 0.35]  # smaller, worse
    param_dict['BW'] = [16, 32, 48, 64, 80]  # larger, worse
    param_dict['GNC'] = [0.001, 0.002, 0.005, 0.01, 0.05]  # larger, worse
    param_dict['GB'] = [7, 9, 13, 17, 21]  # larger, worse
    param_dict['JPEG'] = [2, 3, 4, 5, 6]  # larger, worse

    # level starts from 1, list starts from 0
    return param_dict[dist_type][level - 1]
# Fake is 1, Real is 0


class DeepFakeSet(Dataset):
    def __init__(self, file_list, data_type='train', input_size=224):
        self.input_size = input_size
        self.data_type = data_type
        self.file_list = file_list
        self.transform = create_train_transforms(
            size=input_size) if data_type == 'train' else create_val_transforms(size=input_size)

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = self.transform(image=image)
        img_transformed = data["image"]

        label = img_path.split('/')[-3][0]
        label = 0 if label == "R" else 1

        return img_transformed, label


class DeepFakeSet_celebdf(Dataset):
    def __init__(self, file_list, data_type='train', input_size=224):

        self.input_size = input_size
        self.data_type = data_type
        self.file_list = file_list
        self.transform = create_train_transforms(
            size=input_size) if data_type == 'train' else create_val_transforms(size=input_size)

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = self.transform(image=image)
        img_transformed = data["image"]
        label = img_path.split('/')[-3][-1]
        label = 0 if label == "l" else 1

        return img_transformed, label


class DeepFakeSet_dfdc(Dataset):
    def __init__(self, file_list, data_type='train', input_size=224):

        self.input_size = input_size
        self.data_type = data_type
        self.file_list = file_list
        self.transform = create_train_transforms(
            size=input_size) if data_type == 'train' else create_val_transforms(size=input_size)
        this_dir = '/home/yagami0zero/Shiufu/TA/dataset/txt'
        txt_file = os.path.join(this_dir, 'test-list-dfdc.txt')

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            self.video_paths = [name.strip().split()[1] for name in lines]
            self.labels = [int(name.strip().split()[0]) for name in lines]

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = self.transform(image=image)
        img_transformed = data["image"]
        index = self.video_paths.index(img_path.split('/')[-2])
        label = self.labels[index]
        return img_transformed, label


class DeepFakeSet_pre(Dataset):
    def __init__(self, file_list, data_type='train', input_size=224, dist_type='random', level= 'random'):
        self.input_size = input_size
        self.data_type = data_type
        self.file_list = file_list
        self.transform = create_train_transforms(
            size=input_size) if data_type == 'train' else create_val_transforms(size=input_size)
        if dist_type == 'random':
            dist_types = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG']
            type_id = random.randint(0, 5)
            self.dist_type = dist_types[type_id]
        else:
            self.dist_type = dist_type
             # get distortion level
        if level == 'random':
            self.level = random.randint(1, 5)
        else:
            self.level = int(level)

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        dist_param = get_distortion_parameter(self.dist_type, self.level)
        dist_function = get_distortion_function(self.dist_type)
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = dist_function(image, dist_param)
        img_transformed = self.transform(image=image)['image']

        label = img_path.split('/')[-3][0]
        label = 0 if label == "R" else 1

        return img_transformed, label


# %%
# Video_Transformer
# pip install onnxruntime==1.6.0


class Model(nn.Module):
    def __init__(self, num_classes=2, sequence_length=1):
        super(Model, self).__init__()
        self.model = ImageTransformer('B_16_imagenet1k', pretrained=True, image_size=299, num_classes=2,
                                      seq_embed=True, hybrid=True, device='cuda')
        self.linear = nn.Linear(768, num_classes)

    def forward(self, x):

        x = self.model(x)
        x = self.linear(x)
        return x


model = Model()

# %%
# RFM 224

model = eval('xception')(num_classes=2, pretrained=False).cuda()

# %%
# dummy = torch.rand((1, 3,299,299))
# model = Model()
# model.cuda()
# out = model(dummy.cuda())
# out

# %%
# input_size = 224
# batch_size = 64
# test_batch_size = 64
# train_data = DeepFakeSet(train_list, transform=train_transforms, data_type='train', input_size = input_size)
# valid_data = DeepFakeSet(valid_list, transform=val_transforms, data_type='val', input_size = input_size)
# celeb_data = DeepFakeSet_celebdf(celebdf_list, transform=val_transforms, input_size = input_size)
# dfdc_data = DeepFakeSet_dfdc(dfdc_list, transform=val_transforms, input_size = input_size)
# train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
# eval_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)

# #Inference
# CelebDF_dataloader = DataLoader(celeb_data, batch_size=test_batch_size, shuffle=False, num_workers=4)
# DFDC_dataloader = DataLoader(dfdc_data, batch_size=test_batch_size, shuffle=False, num_workers=4)

# %%
torch.backends.cudnn.benchmark = True

# %%
# Video_Transformer
if __name__ == '__main__':
    root_path = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP'
    batch_size = 64
    test_batch_size = 64
    input_size = 299
    epoch_start = 0
    num_epochs = 100
    device_id = 0  # set the gpu id
    lr = 0.0005
    use_adv = False  # use fake adv
    use_blending = False
    seed = 23
    the_last_loss = 100
    patience = 10
    trigger_times = 0
    sequence_length = 1
    resume_epoch = 0
    dataset = 'FFPP'
    model_name = 'Video_Transformer'
    print('using '+model_name)

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath("__file__")))
    exp_name = os.path.dirname(os.path.abspath("__file__")).split('/')[-1]

    if resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    # save_dir = os.path.join(save_dir_root, 'run', model_name)
    saveName = model_name + '-' + dataset
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime(
        '%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    model_path = None

    if torch.cuda.device_count() > 1:  # 使用多GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    is_train = True
    if is_train:

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.9)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
#         original_dataset = AllDataset(root_path, data_type='train', is_one_hot=False, input_size=input_size, use_adv=True,
#                           use_real_adv=False, use_blending=True, seed=seed, num_classes=2)
        train_data = DeepFakeSet(
            train_list, data_type='train', input_size=input_size)
        valid_data = DeepFakeSet(
            valid_list, data_type='val', input_size=input_size)
        celeb_data = DeepFakeSet_celebdf(
            celebdf_list, data_type='val', input_size=input_size)
        dfdc_data = DeepFakeSet_dfdc(
            dfdc_list, data_type='val', input_size=input_size)
        train_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(
            dataset=valid_data, batch_size=batch_size, shuffle=True)

        # Inference
        CelebDF_dataloader = DataLoader(
            celeb_data, batch_size=test_batch_size, shuffle=False, num_workers=4)
        DFDC_dataloader = DataLoader(
            dfdc_data, batch_size=test_batch_size, shuffle=False, num_workers=4)
        train_loss_record = []
        valid_loss_record = []
        train_acc_record = []
        valid_acc_record = []
        # train
        best_auc = 0.5
        best_auc_dfdc = 0.5
        best_acc = 0
        for epoch in range(epoch_start, num_epochs):
            train_acc, train_loss = train_model(
                model, criterion, optimizer, epoch)
            if epoch % 1 == 0 or epoch == num_epochs - 1:
                auc, val_acc, the_current_loss = eval_model(
                    model, epoch, eval_loader)
                # Inference DFDC_dataloader
                auc_dfdc = Inference(model, DFDC_dataloader, is_tta=False)
                # Early stopping
                print('The current loss:', the_current_loss)
                if the_current_loss > the_last_loss:
                    trigger_times += 1
                    print('trigger times:', trigger_times)
                    # 連續10次才停
                    if trigger_times >= patience:
                        print('Early stopping!')
                        break
                else:
                    print('trigger times: 0')
                    trigger_times = 0
                the_last_loss = the_current_loss
                # Save best auc
                if best_auc < auc:
                    best_auc = auc
                    # torch.save(model.state_dict(), '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc))
                    # torch.save(model.state_dict(), '{}/{}_best_auc.pth'.format(store_name, epoch))
                    torch.save(model.state_dict(), os.path.join(
                        save_dir, 'models', '{}_best_auc.pth'.format(saveName)))
                    print('save best auc {0:.4f} in epoch {1}'.format(
                        best_auc, epoch))
                # Save best auc dfdc
                if best_auc_dfdc < auc_dfdc:
                    best_auc_dfdc = auc_dfdc
                    # torch.save(model.state_dict(), '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc))
                    # torch.save(model.state_dict(), '{}/{}_best_auc.pth'.format(store_name, epoch))
                    torch.save(model.state_dict(), os.path.join(
                        save_dir, 'models', '{}_best_auc_dfdc.pth'.format(saveName)))
                    print('save best auc_dfdc {0:.4f} in epoch {1}'.format(
                        best_auc_dfdc, epoch))
#                 # Save best acc
#                 if best_acc < val_acc:
#                     best_acc = val_acc
#                     #torch.save(model.state_dict(), '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc))
#                     #torch.save(model.state_dict(), '{}/{}_best_auc.pth'.format(store_name, epoch))
#                     torch.save(model.state_dict(), os.path.join(save_dir, 'models', '{}_best_acc.pth'.format(saveName)))
#                     print('save best acc {0:.4f} in epoch {1}'.format(best_acc, epoch))

                train_loss_record.append(train_loss)
                valid_loss_record.append(the_current_loss)
                train_acc_record.append(train_acc)
                valid_acc_record.append(val_acc)

            print('current best auc:', best_auc)  # AUC
            print('current best auc in dfdc:', best_auc_dfdc)  # best auc dfdc
        save_loss(train_loss_record, valid_loss_record, save_dir)
        save_acc(train_acc_record, valid_acc_record, save_dir)
#         print('evaluate test data')
#         Inference(model, test_dataloader)
        print('evaluate CelebDF in ffpp')
        Inference(model, CelebDF_dataloader)
        print('evaluate DFDC in ffpp')
        Inference(model, DFDC_dataloader)
#     else:
#         input_size = 300
#         seed = 2021
#         batch_size = 64
#         start = time.time()
#         epoch_start = 1
#         num_epochs = 1
#         xdl_test = DFGCDataset(root_path, data_type='test', input_size=input_size, test_adv=False, seed=seed, num_classes=2)
#         test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=True, num_workers=4)
#         test_dataset_len = len(xdl_test)
#         print('test_dataset_len:', test_dataset_len)
#         eval_model(model, epoch_start, test_loader, is_save=False, is_tta=False, metric_name='acc')
#         print('Total time:', time.time() - start)

# %%
train_dir_real = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/RealFF'
train_dir_fake = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/FakeFF/FaceSwap/'
train_dir_fake_2 = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/FakeFF/Face2Face/'
train_dir_fake_3 = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/FakeFF/Deepfakes/'
train_dir_fake_4 = '/home/yagami0zero/Shiufu/TA/data_preparation/data_structure/new_FFPP/FakeFF/NeuralTextures/'

train_list_real = glob.glob(os.path.join(train_dir_real, '*', '*'))
train_list_fake = glob.glob(os.path.join(train_dir_fake, '*', '*'))
train_list_fake_2 = glob.glob(os.path.join(train_dir_fake_2, '*', '*'))
train_list_fake_3 = glob.glob(os.path.join(train_dir_fake_3, '*', '*'))
train_list_fake_4 = glob.glob(os.path.join(train_dir_fake_4, '*', '*'))

train_list_real, valid_list_real = train_test_split(
    train_list_real, random_state=seed, train_size=0.6)
train_list_fake, valid_list_fake = train_test_split(
    train_list_fake, random_state=seed, train_size=0.6)
train_list_fake_2, valid_list_fake_2 = train_test_split(
    train_list_fake_2, random_state=seed, train_size=0.6)
train_list_fake_3, valid_list_fake_3 = train_test_split(
    train_list_fake_3, random_state=seed, train_size=0.6)
train_list_fake_4, valid_list_fake_4 = train_test_split(
    train_list_fake_4, random_state=seed, train_size=0.6)

np.random.shuffle(train_list_real)
np.random.shuffle(train_list_fake)
np.random.shuffle(train_list_fake_2)
np.random.shuffle(train_list_fake_3)
np.random.shuffle(train_list_fake_4)
np.random.shuffle(train_list_real)
np.random.shuffle(train_list_fake)
np.random.shuffle(train_list_fake_2)
np.random.shuffle(train_list_fake_3)
np.random.shuffle(train_list_fake_4)


np.random.shuffle(valid_list_real)
np.random.shuffle(valid_list_fake)
np.random.shuffle(valid_list_fake_2)
np.random.shuffle(valid_list_fake_3)
np.random.shuffle(valid_list_fake_4)
np.random.shuffle(valid_list_real)
np.random.shuffle(valid_list_fake)
np.random.shuffle(valid_list_fake_2)
np.random.shuffle(valid_list_fake_3)
np.random.shuffle(valid_list_fake_4)

train_list = []
train_list.extend(train_list_real[:100000])
train_list.extend(train_list_fake[:25000])
train_list.extend(train_list_fake_2[:25000])
train_list.extend(train_list_fake_3[:25000])
train_list.extend(train_list_fake_4[:25000])

valid_list = []
valid_list.extend(valid_list_real[:20000])
valid_list.extend(valid_list_fake[:5000])
valid_list.extend(valid_list_fake_2[:5000])
valid_list.extend(valid_list_fake_3[:5000])
valid_list.extend(valid_list_fake_4[:5000])

print(f"Train Data Real: {len(train_list_real)}")
print(f"Train Data Fake: {len(train_list_fake)}")
print(f"Train Data Fake 2: {len(train_list_fake_2)}")
print(f"Train Data Fake 3: {len(train_list_fake_3)}")
print(f"Train Data Fake 4: {len(train_list_fake_4)}")

np.random.shuffle(train_list)
np.random.shuffle(train_list)
np.random.shuffle(train_list)
np.random.shuffle(train_list)

np.random.shuffle(valid_list)
np.random.shuffle(valid_list)
np.random.shuffle(valid_list)
np.random.shuffle(valid_list)


def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        HorizontalFlip(),
        GaussNoise(p=0.1),
        GaussianBlur(p=0.1),
        RandomRotate90(),
        Resize(height=size, width=size),
        PadIfNeeded(min_height=size, min_width=size,
                    border_mode=cv2.BORDER_CONSTANT),
        RandomCrop(height=size, width=size),
        OneOf([RandomBrightnessContrast(), FancyPCA(),
              HueSaturationValue()], p=0.5),
        OneOf([CoarseDropout(), GridDropout()], p=0.5),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                         rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )


def create_val_transforms(size=300):
    return Compose([
        Resize(height=size, width=size),
        # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        # GaussianBlur(blur_limit=3, p=1),
        # CenterCrop(height=size, width=size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class DeepFakeSet(Dataset):
    def __init__(self, file_list, data_type='train', input_size=224):
        self.input_size = input_size
        self.data_type = data_type
        self.file_list = file_list
        self.transform = create_train_transforms(
            size=input_size) if data_type == 'train' else create_val_transforms(size=input_size)

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = self.transform(image=image)
        img_transformed = data["image"]

        label = img_path.split('/')[-3][0]
        label = 0 if label == "R" else 1

        return img_transformed, label


# %%
# #查best 模型的其他數據
# device_id = 0
# input_size = 224
# batch_size = 256
# train_data = DeepFakeSet(train_list, data_type='train', input_size = input_size)
# valid_data = DeepFakeSet(valid_list, data_type='val', input_size = input_size)
# celeb_data = DeepFakeSet_celebdf(celebdf_list, data_type='val', input_size = input_size)
# dfdc_data = DeepFakeSet_dfdc(dfdc_list, data_type='val', input_size = input_size)
# train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
# eval_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)

# #Inference
# CelebDF_dataloader = DataLoader(celeb_data, batch_size=batch_size, shuffle=False, num_workers=4)
# DFDC_dataloader = DataLoader(dfdc_data, batch_size=batch_size, shuffle=False, num_workers=4)
# from RFM.utils1.utils import data_prefetcher_two, cal_fam, setup_seed, calRes
# from pretrainedmodels import xception

# model = eval('xception')(num_classes=2, pretrained=False).cuda()
# model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
# pretrained_dict= torch.load(model_path2)
# pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}

# model.load_state_dict(pretrained_dict)
# model.cuda()
# Inference(model,eval_loader, is_tta=False)

# %%

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CS', level='1')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CS', level='2')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CS', level='3')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CS', level='4')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CS', level='5')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CC', level='1')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CC', level='2')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CC', level='3')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CC', level='4')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='CC', level='5')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='BW', level='1')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='BW', level='2')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='BW', level='3')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='BW', level='4')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='BW', level='5')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GNC', level='1')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GNC', level='2')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GNC', level='3')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GNC', level='4')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GNC', level='5')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GB', level='1')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GB', level='2')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GB', level='3')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GB', level='4')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='GB', level='5')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='JPEG', level='1')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='JPEG', level='2')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='JPEG', level='3')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='JPEG', level='4')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%
# perturbations
device_id = 0
input_size = 224
test_dataset = DeepFakeSet_pre(valid_list, data_type='val', input_size=input_size,
                               dist_type='JPEG', level='5')
test_dataloader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=4)
model = eval('xception')(num_classes=2, pretrained=False).cuda()
model_path2 = './run/run_1/models/RFM-FFPP_best_auc.pth'
pretrained_dict = torch.load(model_path2)
pretrained_dict = {key.replace(
    "module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.cuda()
Inference(model, test_dataloader, is_tta=False)

# %%

# %%

# %%
