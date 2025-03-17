from model import LitNeuralNet
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import dataset
import pandas as pd



transform_test = A.Compose([
                A.Resize(224, 224),
                A.ImageCompression(quality_lower=15, quality_upper=30, p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
        ])

data = dataset.DatasetPred(path_file="../data/global_test.csv", transforms=transform_test)
data_loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=1)
model = LitNeuralNet.load_from_checkpoint("/workspace/scripts/lightning_logs/version_64/checkpoints/epoch=2-step=10794.ckpt")
trainer = Trainer()
logits = trainer.predict(model, data_loader)
logits = torch.cat(logits)
probs = torch.sigmoid(logits)

predictions = (probs > 0.5).int()
predictions = predictions.tolist()
predictions = [predict[0] for predict in predictions]

predictions_df = pd.read_csv("../data/global_test.csv", sep=",")
predictions_df["label"] = predictions

predictions_df.to_csv('../data/Submissions.csv', sep=',', index = False)


