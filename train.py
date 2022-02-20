import numpy as np
import torch
import torchvision
from torch import nn
from torch import squeeze
import torch.optim as optim
import torch.nn.functional as F
from Unet import Unet
from dataset import WMH_Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
#from IOULoss import IoULoss


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
LEARNING_RATE = 1e-4
Batch_size = 32
NUM_EPOCH = 100
DEVICE = device
Train_Img_Dir = "/home/yuanzhe/Desktop/unet_segmentation/slice_result/new_data/pred/"
Train_Mask_Dir = "/home/yuanzhe/Desktop/unet_segmentation/slice_result/new_data/g_truth/"


def get_loaders(train_dir, 
                train_mask_dir, 
                batch_size,
                train_transform,
                ):
    full_dataset = WMH_Dataset(image_dir = train_dir, mask_dir = train_mask_dir, transform = train_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print(f"test dataset {test_dataset}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader

def IoULoss(inputs, targets, smooth = 1e-6):
    #comment out if your model contains a sigmoid or equivalent activation layer\
    inputs = torch.sigmoid(inputs)       
        
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
        
    IoU = (intersection + smooth)/(union + smooth)
    #print(f"IoU Loss {IoU}")
    return 1 - IoU

def train(loader, model, optimizer, loss_fn, scaler):
    model.train()
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)
        data = np.swapaxes(data, 2, 3)
        targets = squeeze(targets, 1)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #plt.imshow(predictions, cmap = 'gray')
            #plt.show()
            #loss = loss_fn(predictions, targets)
            loss = IoULoss(predictions, targets)
            #print(f"loss {loss}")

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        '''
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")'''

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = np.swapaxes(y, 2, 3)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            '''preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()'''
            preds = model(x)
            print(f"x shape {preds.shape}")
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        print(f"y shape {y.shape}")
        y = np.swapaxes(y, 2, 3)
        torchvision.utils.save_image(y, f"{folder}{idx}.png")
        preds = (preds > 0.5).float()
        num_correct += (preds == y).sum()
        num_pixels += torch.numel(preds)
        dice_score += (2 * (preds * y).sum()) / (
            (preds + y).sum() + 1e-8
        )

    print(
        f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"
    )

    model.train()

model = Unet(3, 2)
model = model.to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.2)),
])

train_loader, val_loader = get_loaders(
    Train_Img_Dir,
    Train_Mask_Dir,
    Batch_size,
    train_transform = None, 
)

scaler = torch.cuda.amp.GradScaler()
for epochs in range(NUM_EPOCH):
    print(f"\nEpoch {epochs+1}\n -------------------------------")
    print("Training...")
    train(train_loader, model, optimizer, loss_fn, scaler)
    check_accuracy(val_loader, model, device = DEVICE)
print("Training Done!\n")
save_predictions_as_imgs(val_loader, model, folder = "/home/yuanzhe/Desktop/unet_segmentation/2D_UNET/")
#test(val_loader, model)
torch.save(model, "2d_unet.pth")