import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import torch
import torch.nn as nn

def readImage(path,subjN):
    img_array = []
    img_dir = []
    counter = 0
    for directory_path in glob.glob(path):
        img_dir.append(directory_path)
    img_dir.sort()
    for path_i in img_dir:
        img = nib.load(path_i)
        img = nib.casting.int_to_float(np.asanyarray(img.dataobj), np.int16)
        if np.shape(img) == (224,256,48):
            img = img / np.max(img)
            img_array.append(img)
            counter += 1
        if counter >= subjN: break
    img_array = np.array(img_array)
    print("Image reading Done\n")
    return img_array

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def sliceImage(img):
    print(img.shape)
    n, row, col,z = np.shape(img)
    img_stack = np.swapaxes(img, 0, 1)
    img_stack = np.swapaxes(img_stack, 1, 2)
    img_stack = np.reshape(img_stack, (row, col, n*z))
    print("Image slicing Done\n")
    return img_stack

def splitTrainTest(img_stack):
    # Split our img paths into a training and a validation set
    row, col, num = np.shape(img_stack)
    val_samples = round(0.1 * num)
    np.random.seed(42)
    col_idx = np.random.permutation(img_stack.shape[2])
    img_stack = img_stack[:,:,col_idx]
    train_img_stack = img_stack[:,:,0:num-val_samples]
    val_img_stack = img_stack[:,:,num-val_samples-1:-1]

    train_img_stack = np.swapaxes(train_img_stack, 2,0)
    train_img_stack = np.swapaxes(train_img_stack, 2,1)
    #train_img_stack = tf.cast(train_img_stack, tf.float32)
    print(train_img_stack.dtype)
    train_img_stack = train_img_stack.type(torch.FloatTensor)
    print(train_img_stack.dtype)

    val_img_stack = np.swapaxes(val_img_stack, 2,0)
    val_img_stack = np.swapaxes(val_img_stack, 2,1)
    #val_img_stack = tf.cast(val_img_stack, tf.float32)
    print(val_img_stack.dtype)
    val_img_stack = val_img_stack.type(torch.FloatTensor)
    print(val_img_stack.dtype)

    print("Data set split done\n")
    return train_img_stack, val_img_stack

## Reading T1, T2 and flair images
#T2_img = readImage("/home/yuanzhe/Desktop/unet_segmentation/wmh_data/data/90*/80*/step12_WM/T2/*_T2.nii", 100)
FLAIR_img = readImage("/home/yuanzhe/Desktop/unet_segmentation/wmh_data/data/90*/80*/step12_WM/T2/r*_FLAIR.nii", 100)
#T1_img = readImage("/home/yuanzhe/Desktop/unet_segmentation/wmh_data/data/90*/80*/step12_WM/T2_2_T1/r*_T1.nii", 100)
mask = readImage("/home/yuanzhe/Desktop/unet_segmentation/wmh_data/data/90*/80*/step12_WM/T2/r*_FLAIR_wmh.nii", 100)
## Slicing imases
#T2_stack = sliceImage(T2_img)
#T1_stack = sliceImage(T1_img)
FLAIR_stack = sliceImage(FLAIR_img)
mask_stack = sliceImage(mask)
## Reading unseen images
#unseen_T1 = readImage("/home/yuanzhe/Desktop/unet_segmentation/wmh_data/unseen_imgs/90*/80*/step12_WM/T2_2_T1/r*_T1.nii", 5)
#print(unseen_T1.shape)
#unseen_T2 = readImage("/home/yuanzhe/Desktop/unet_segmentation/wmh_data/unseen_imgs/90*/80*/step12_WM/T2/*_T2.nii", 5)
unseen_FLAIR = readImage("/home/yuanzhe/Desktop/unet_segmentation/wmh_data/unseen_imgs/90*/80*/step12_WM/T2/r*_FLAIR.nii", 5)
unseen_mask = readImage("/home/yuanzhe/Desktop/unet_segmentation/wmh_data/unseen_imgs/90*/80*/step12_WM/T2/r*_FLAIR_wmh.nii", 5)
## Slicing unseen images
#unseen_T1 = sliceImage(unseen_T1)
#unseen_T2 = sliceImage(unseen_T2)
unseen_FLAIR = sliceImage(unseen_FLAIR)
#unseen_mask = sliceImage(unseen_mask)
#unseen_img =  np.stack((unseen_T2, unseen_T1, unseen_FLAIR), axis = 3)
unseen_mask = np.stack((unseen_mask, unseen_mask, unseen_mask), axis = 3)
#unseen_T1 = np.stack((unseen_T1, unseen_T1, unseen_T1), axis = 3)
#unseen_T2 = np.stack((unseen_T2, unseen_T2, unseen_T2), axis = 3)
unseen_FLAIR = np.stack((unseen_FLAIR, unseen_FLAIR, unseen_FLAIR), axis = 3)
#print(unseen_T1.shape)
## Splitting training and testing images
#train_T2, test_T2 = splitTrainTest(T2_stack)
#train_T1, test_T1 = splitTrainTest(T1_stack)
train_FLAIR, test_FLAIR = splitTrainTest(FLAIR_stack)
train_mask, test_mask = splitTrainTest(mask_stack)

print(train_FLAIR)
print(train_FLAIR.shape)


#train_img = np.stack((train_T2, train_T1, train_FLAIR), axis = 3)
#test_img = np.stack((test_T2, test_T1, test_FLAIR), axis = 3)
#train_mask = np.stack((train_mask, train_mask, train_mask), axis = 3)
#test_mask = np.stack((test_mask, test_mask, test_mask), axis = 3)

# plt.figure(figsize=(20, 15))
# plt.subplot(1,3,1)
# plt.imshow(train_T2[220,:,:], cmap="gray")
# plt.imshow(train_mask[220,:,:],cmap="gray",alpha=0.5)
# plt.gca().get_xaxis().set_visible(False)
# plt.gca().get_yaxis().set_visible(False)
# plt.subplot(1,3,2)
# plt.imshow(train_T1[220,:,:], cmap="gray")
# plt.imshow(train_mask[220,:,:],cmap="gray",alpha=0.5)
# plt.gca().get_xaxis().set_visible(False)
# plt.gca().get_yaxis().set_visible(False)
# plt.subplot(1,3,3)
# plt.imshow(train_FLAIR[220,:,:], cmap="gray")
# plt.imshow(train_mask[220,:,:],cmap="gray",alpha=0.5)
# plt.gca().get_xaxis().set_visible(False)
# plt.gca().get_yaxis().set_visible(False)
# plt.savefig("dummy_name.png")

print("Data preprocessing done!")