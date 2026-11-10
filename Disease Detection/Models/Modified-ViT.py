import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

import timm
from timm.loss import LabelSmoothingCrossEntropy

import sys
import os
import numpy as np
from tqdm import tqdm
import time
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Train the DeiT-Tiny disease classifier.")
parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "Dataset" / "Disease Detection Dataset", help="Directory containing train, val, and test splits")
parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=128, help="Training and evaluation batch size")
args = parser.parse_args()

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

def get_data_loaders(data_dir, batch_size, train = False):
    if train:
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.RandomErasing(p=0.2, value='random')
        ])
        
        train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        
        return train_loader, len(train_data)
    
    else:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        val_data = datasets.ImageFolder(os.path.join(data_dir, "val/"), transform=transform)
        test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return val_loader, test_loader, len(val_data), len(test_data)
    
dataset_path = args.dataset_dir

(train_loader, train_data_len) = get_data_loaders(dataset_path, args.batch_size, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, args.batch_size, train=False)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train": train_data_len,
    "val": valid_data_len
}

classes = get_classes(dataset_path / "train")
print(classes, len(classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

model = timm.create_model('deit_tiny_patch16_224', pretrained=True)

for param in model.parameters():
    param.requires_grad = False

n_inputs = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))
)
model = model.to(device)
print(model.head)

criterion = LabelSmoothingCrossEntropy()
criterion = criterion.to(device)
optimizer = optim.Adam(model.head.parameters(), lr=0.001)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc =  running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Val Acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model



model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs)



model_ft.eval()

test_loss = 0.0
y_pred = []
y_true = []
for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
    test_loss += loss.item() * inputs.size(0)
    y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
    y_true.extend(labels.cpu().numpy())

test_loss /= test_data_len
test_accuracy = np.mean(np.asarray(y_pred) == np.asarray(y_true))
print(f"Test Loss: {test_loss:.4f}; Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

cf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)), normalize='true')
df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
plt.figure(figsize = (12, 8))
sns.heatmap(df_cm, annot=True)
plt.savefig(Path(__file__).with_name("modified_vit_confusion_matrix.png"), dpi=100, bbox_inches='tight')
torch.save(model_ft.state_dict(), Path(__file__).with_name("modified_vit_model.pth"))