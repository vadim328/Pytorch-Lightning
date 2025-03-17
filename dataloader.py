import pytorch_lightning as pl
from torch.utils.data import DataLoader



class MyDataModule(pl.LightningDataModule):
	def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
		super().__init__()
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset
		self.batch_size = batch_size

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
