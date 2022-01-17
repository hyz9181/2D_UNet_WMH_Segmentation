import torch
from torch import nn
import torch.optim as optim
from Unet import Unet
from dataset import WMH_Dataset
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
LEARNING_RATE = 1e-4
Batch_size = 16
NUM_EPOCH = 5
DEVICE = device
Train_Img_Dir = "/home/yuanzhe/Desktop/unet_segmentation/result/pred/"
Train_Mask_Dir = "/home/yuanzhe/Desktop/unet_segmentation/result/g_truth/"

def get_loaders(train_dir, 
                train_mask_dir, 
                batch_size,
                train_transform,
                ):
    full_dataset = WMH_Dataset(image_dir = train_dir, mask_dir = train_mask_dir, transform = train_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

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

def train(loader, model, optimizer, loss_fn, scaler):
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

model = Unet(3, 2)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

train_loader, val_loader = get_loaders(
    Train_Img_Dir,
    Train_Mask_Dir,
    Batch_size,
    train_transform = None, 
)

scaler = torch.cuda.amp.GradScaler()
for epochs in range(NUM_EPOCH):
    print("epoch ")
    print(epochs)
    train(train_loader, model, optimizer, loss_fn, scaler)
print("Training Done!s")