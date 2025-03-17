from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import vit_b_16
from torchvision.models import swin_t
from torch.nn import functional as F
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch import nn
import torch
from torchmetrics.classification import BinaryAccuracy


class LitNeuralNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = swin_t(weights=None)        
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features=768, out_features=1, bias=True)

        self.accuracy = BinaryAccuracy()
        self.test_predictions = []
        self.acc_score = None
        self.loss = None
 
    def forward(self, image):
        img_out = self.model(image)
        return img_out

    def training_step(self, batch, batch_idx):
        image, y = batch
        #y = torch.tensor(y)
        y = y.unsqueeze(1)
        y_hat = self(image)

        self.loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        
        # compute accuracy
        y_prob = torch.sigmoid(y_hat)
        y_pred = (y_prob > 0.5).float()
        self.acc_score = self.accuracy(y_pred, y)

        self.log('train_loss_step', self.loss, on_step=True, on_epoch=False)
        self.log('train_accuracy_step', self.acc_score, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', self.loss, on_step=False, on_epoch=True)
        self.log('train_accuracy_epoch', self.acc_score, on_step=False, on_epoch=True)
        return self.loss

    def validation_step(self, batch, batch_idx):
        image, y = batch
        #y = torch.tensor(y)
        y = y.unsqueeze(1)
        y_hat = self(image)

        self.loss = F.binary_cross_entropy_with_logits(y_hat, y.float())

        # compute accuracy
        y_prob = torch.sigmoid(y_hat)
        y_pred = (y_prob > 0.5).float()
        self.acc_score = self.accuracy(y_pred, y)

        self.log('validation_loss', self.loss)
        self.log('validation_accuracy', self.acc_score)

    def test_step(self, batch, batch_idx):
        image, y = batch
        #y = torch.tensor(y)
        y = y.unsqueeze(1)
        y_hat = self(image)

        self.loss = F.binary_cross_entropy_with_logits(y_hat, y.float())

        # compute accuracy
        y_prob = torch.sigmoid(y_hat)
        y_pred = (y_prob > 0.5).float()
        self.test_predictions.append(y_pred)
        self.acc_score = self.accuracy(y_pred, y)
        
        self.log('test_loss', self.loss)
        self.log('test_accuracy', self.acc_score)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch
        return self(image)
    
    def on_train_epoch_end(self):
        print("epoch accuracy: ", self.acc_score.item())
        print("epoch loss: ", self.loss.item())

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_predictions, dim=0)
        return all_preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
