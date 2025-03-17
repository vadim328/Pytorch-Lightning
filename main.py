import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

import dataset
from argparse import ArgumentParser

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import LitNeuralNet

from split_dataset import train_val_test_split
from dataloader import MyDataModule



def main(hparams):

	if bool(hparams.collect_dataset):
		train_val_test_split()
		print("---------------\nCollect Dataset successful \n---------------")

	transform_train = A.Compose([
		A.Resize(224, 224),
		A.RandomRotate90(p=0.25),
		A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, interpolation=1, p=0.5),
		#A.cutout(max_h_size=32, max_w_size=32, p=0.05),
		A.CLAHE(clip_limit=2, p=0.1),
		A.RandomBrightnessContrast(p=0.2),
		A.GaussNoise(p=0.2),
		A.MotionBlur(blur_limit=3, p=0.2),
		A.ISONoise(p=0.2),
		A.ImageCompression(quality_lower=15, quality_upper=30, p=0.25),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2()
	])

	transform_test = A.Compose([
		A.Resize(224, 224),
		A.ImageCompression(quality_lower=15, quality_upper=30, p=0.25),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2()
	])

	train_data = dataset.DatasetTrain(path_file="../data/train.csv", transforms=transform_train)
	print("---------------\nTrain Dataset loaded \n---------------")
	
	val_data = dataset.DatasetVal(path_file="../data/val.csv", transforms=transform_test)
	print("---------------\nValidation Dataset loaded \n---------------")

	test_data = dataset.DatasetTest(path_file="../data/test.csv", transforms=transform_test)
	print("---------------\nTest Dataset loaded \n----------------")
	
	my_data_module = MyDataModule(train_data, val_data, test_data, batch_size=16)

	model_x = LitNeuralNet()
	trainer = pl.Trainer(max_epochs=int(hparams.epochs), accelerator=hparams.accelerator, \
		devices=int(hparams.devices), check_val_every_n_epoch=1)
	trainer.fit(model_x, datamodule=my_data_module)

	test_data_loader = my_data_module.test_dataloader()
	trainer.test(dataloaders=test_data_loader)
	print(model_x.on_test_epoch_end())


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--accelerator", default="cuda")
	parser.add_argument("--devices", default=1)
	parser.add_argument("--epochs", default=5)
	parser.add_argument("--collect_dataset", default=False)
	args = parser.parse_args()

	main(args)
