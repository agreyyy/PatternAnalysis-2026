from dataset import TrainData, TestData, ValidationData
from modules import BaselineModel, PaperModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import sys
import time

print("ARG 1 MUST BE PATH TO THE DATASET")
PROSTATE_LABEL = (0,0,0,1)
NUM_CLASSES = 4
BASIC_EPOCHS = 3
PAPER_EPOCHS = 3


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask, output):
        loss = 0.0
        epsilon = 1e-6
        for k in range(NUM_CLASSES):
            p = output[:, k, :, :]
            y = mask[:, k, :, :]

            p_flat = p.contiguous().view(-1)
            y_flat = y.contiguous().view(-1)
            intersection = torch.sum(p_flat * y_flat)
            total = torch.sum(p_flat) + torch.sum(y_flat)

            dice_coef = (2.0 * intersection + epsilon) / (total + epsilon)
            loss += (1.0 - dice_coef)

        return loss/NUM_CLASSES

class BasicUNet: 
    def __init__(self, device, data_path):
        self.inference = []
        self.vram_usage = []
        self.model = BaselineModel().to(device)
        self.dev = device
        training = TrainData(data_path)
        testing = TestData(data_path)
        validation = ValidationData(data_path)
        self.train = DataLoader(
            dataset=training,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.test = DataLoader(
            dataset=testing,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.validate = DataLoader(
            dataset=validation,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )


    def train_model(self):
        self.model.train()
        device = self.dev
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        loss_fn = DiceLoss()

        for epoch in range(BASIC_EPOCHS):
            start = time.time()
            for i, (mask, image) in enumerate(self.train):
                print(f"{i} iterations")
                mask = mask.to(device)
                image = image.to(device)
                t = time.time()
                output = self.model(image)
                inference_time = time.time() - t
                loss = loss_fn(mask, output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.inference.append(inference_time)
                self.vram_usage.append(torch.cuda.memory_allocated())
            print(f"EPOCH COMPLETED!: {start - time.time()}s")        

dev = torch.device('cuda' if torch.cuda.is_available() else "cpu")
unet = BasicUNet(dev, sys.argv[1])
unet.train_model()

def improved_unet_train(dev, train):
    model = PaperModel(NUM_CLASSES)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
