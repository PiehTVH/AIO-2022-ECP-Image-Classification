import os

import cv2
import pandas as pd
import json
import torch.utils.data as data
import torch
from PIL import Image


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError("Failed to read {}".format(image_file))
    return img


class Product10KDataset(data.Dataset):
    def __init__(
        self, root, annotation_file, convert_file, transforms, is_inference=False, with_bbox=False
    ):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.cls_convert = json.load(open(convert_file, 'r'))
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(0)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, target, _ = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.is_inference:
            return img
        else:
            return img, self.cls_convert[str(target)]

    def __len__(self):
        return len(self.imlist)


class Product10KFS(data.Dataset):
    def __init__(
        self, root, annotation_file, convert_file, transforms, is_inference=False, is_val=False, n_pair=None, with_bbox=False
    ):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.cls_conv = json.load(open(convert_file, 'r'))
        self.transforms = transforms
        self.is_inference = is_inference
        self.is_val = is_val
        self.n_pair = n_pair
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(0)

        if self.is_inference:
            impath, _ = self.imlist.iloc[index]
        else:
            impath, target = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)

        if not self.is_val and not self.is_inference:
            pos_samp = self.imlist[self.imlist['class']==target].sample(n=1)['name'].values[0]
            pos_img = read_image(os.path.join(self.root, pos_samp))
            pos_img = Image.fromarray(pos_img)
            pos_img = self.transforms(pos_img)
            neg_samps = list(self.imlist[self.imlist['class']!=target].sample(n=self.n_pair)['name'].values)
            neg_imgs = [read_image(os.path.join(self.root, neg_samp)) for neg_samp in neg_samps]
            neg_imgs = [Image.fromarray(neg_img) for neg_img in neg_imgs]
            neg_imgs = [self.transforms(neg_img) for neg_img in neg_imgs]
            samps = torch.stack([pos_img] + neg_imgs, dim=0)

        if self.is_inference:
            return img
        elif self.is_val:
            return img, self.cls_conv[str(target)]
        else:
            return img, self.cls_conv[str(target)], samps

    def __len__(self):
        return len(self.imlist)


class Product10KDR(data.Dataset):
    def __init__(
        self, root, annotation_file, transforms, is_inference=False, n_pair=None, with_bbox=False
    ):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.n_pair = n_pair
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(0)

        if self.is_inference:
            impath, _ = self.imlist.iloc[index]
        else:
            impath, target = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)

        if not self.is_inference:
            pos_samp = self.imlist[self.imlist['class']==target].sample(n=1)['name'].values[0]
            pos_img = read_image(os.path.join(self.root, pos_samp))
            pos_img = Image.fromarray(pos_img)
            pos_img = self.transforms(pos_img)
            neg_samps = list(self.imlist[self.imlist['class']!=target].sample(n=self.n_pair)['name'].values)
            neg_imgs = [read_image(os.path.join(self.root, neg_samp)) for neg_samp in neg_samps]
            neg_imgs = [Image.fromarray(neg_img) for neg_img in neg_imgs]
            neg_imgs = [self.transforms(neg_img) for neg_img in neg_imgs]
            samps = torch.stack([pos_img] + neg_imgs, dim=0)

        if self.is_inference:
            return img
        else:
            return img, samps

    def __len__(self):
        return len(self.imlist)


class SubmissionDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        full_imname = os.path.join(self.root, self.imlist["img_path"][index])
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imlist)
