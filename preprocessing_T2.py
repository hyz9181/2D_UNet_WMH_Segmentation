import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import tensorflow as tf


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
    train_img_stack = tf.cast(train_img_stack, tf.float32)

    val_img_stack = np.swapaxes(val_img_stack, 2,0)
    val_img_stack = np.swapaxes(val_img_stack, 2,1)
    val_img_stack = tf.cast(val_img_stack, tf.float32)

    print("Data set split done\n")
    return train_img_stack, val_img_stack


unseen_T2 = readImage("/ocean/projects/med200001p/jlip/unseen_images/T2/*_T2.nii",100)
unseen_mask = readImage("/ocean/projects/med200001p/jlip/unseen_images/T2/r*_wmh.nii", 100)
T2_img = readImage("/ocean/projects/med200001p/jlip/T2/*_T2.nii",100)
mask = readImage("/ocean/projects/med200001p/jlip/T2/r*_FLAIR_wmh.nii",100)
T2_stack = sliceImage(T2_img)
mask_stack = sliceImage(mask)
train_T2, test_T2 = splitTrainTest(T2_stack)
train_mask, test_mask = splitTrainTest(mask_stack)


# plt.figure(figsize=(20, 15))
# for i in range(12):
#     ax = plt.subplot(2,6,i+1)
#     plt.imshow(train_FLAIR[:,:,i+15],cmap = "gray")
#     plt.imshow(train_mask[:,:,i+15],cmap="gray",alpha=0.5)
#     plt.gca().get_xaxis().set_visible(False)
#     plt.gca().get_yaxis().set_visible(False)
# plt.savefig("dummy_name.png")

print("Data preprocessing done!")