import json
from typing import Dict, List, Any
import cv2
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from ExtractFeatures import compute_hue_saturation_brightness, extract_dft_features
from ExtractFeatures import compute_contrast, compute_exposure, compute_sharpness


class Dataset(Dataset):
    def __init__(
            self,
            path_file: str,
    ):
        self.path_file = path_file

    def load_datast(self):
        """optional"""
        image_paths, labels = [], []
        data_frame = pd.read_csv(self.path_file, sep=',')
        image_paths = list(data_frame['file_name'])
        labels = list(data_frame['label'])
        labels = torch.tensor(labels)
        return image_paths, labels
    

class DatasetTrain(Dataset):
    def __init__(
        self,
        path_file: str,
        transforms: List[Dict[str, Any]] = None,
        transform_params: Dict[str, Any] = {},
        name: str = None,
        infinite: bool = False,
    ):
        super().__init__(path_file)
        self.image_paths, self.labels = self.load_datast()
        
        self.transforms = transforms if transforms is not None else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if self.transforms:
            image = self.transforms(image=image)["image"]
	
        return image, label

    def get_class_name(self, label):
        return self.label_encoder.inverse_transform([label])[0]


class DatasetVal(Dataset):
    def __init__(
        self,
        path_file: str,
        transforms: List[Dict[str, Any]] = None,
        transform_params: Dict[str, Any] = {},
        name: str = None,
        infinite: bool = False,
    ):
        super().__init__(path_file)
        self.image_paths, self.labels = self.load_datast()
        
        self.transforms = transforms if transforms is not None else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.transforms:
            image = self.transforms(image=image)["image"]
       
        return image, label

    def get_class_name(self, label):
        return self.label_encoder.inverse_transform([label])[0]


class DatasetTest(Dataset):
    def __init__(
        self,
        path_file: str,
        transforms: List[Dict[str, Any]] = None,
        transform_params: Dict[str, Any] = {},
        name: str = None,
        infinite: bool = False,
    ):
        super().__init__(path_file)
        self.image_paths, self.labels = self.load_datast()

        self.transforms = transforms if transforms is not None else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, label

    def get_class_name(self, label):
        return self.label_encoder.inverse_transform([label])[0]


class DatasetPred(Dataset):
    def __init__(
        self,
        path_file: str,
        transforms: List[Dict[str, Any]] = None,
        transform_params: Dict[str, Any] = {},
        name: str = None,
        infinite: bool = False,
    ):
        self.path_file = path_file        
        self.image_paths = self.load_datast_pred()

        self.transforms = transforms if transforms is not None else None

    def load_datast_pred(self):
        data_frame = pd.read_csv(self.path_file, sep=',')
        data_frame['id'] = '/workspace/data/' + data_frame['id']
        image_paths = list(data_frame['id'])
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image

    def get_class_name(self, label):
        return self.label_encoder.inverse_transform([label])[0]
