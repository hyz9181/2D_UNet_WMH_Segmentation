import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import csv
import sys
import array as arr
from PIL import Image

study_dir = "/home/yuanzhe/Desktop/unet_segmentation/wmh_data/new_data/T2/"
subjectlist_dir = "/home/yuanzhe/Desktop/unet_segmentation/2D_UNET/"
#save_dir = "/home/yuanzhe/Desktop/unet_segmentation/slice_result/new_data/pred/"

def read_from_list(list_name):
	f = open(list_name, 'r')
	f_lines = f.readlines()
	return f_lines

def slice_save(scanID):
    scanID = int(scanID)
    scanID = str(scanID)
    scan_path = study_dir + scanID + "_FLAIR_wmh.nii"
    print(scan_path)
    flair_img = nib.load(scan_path)
    flair_data = flair_img.get_fdata()
    for i in range(18, 26):
        slice = flair_data[:, :, i]
        id = str(i)
        save_name = scanID + "_" + id + "_FLAIR_wmh.jpg"
        print(slice.shape)
        #slice = np.swapaxes(slice, 0, 1)
        #print(slice.shape)
        plt.imshow(slice, cmap = 'gray')
        plt.imsave(save_name, slice, cmap = 'gray')
        #plt.show()

# Main Function

subjectlist_name = str(sys.argv[1])
subjectlist = subjectlist_dir + subjectlist_name
ids = read_from_list(subjectlist)

for i in ids:
    scanID = i
    scan_path = study_dir + i
    print('Beginning ' + i + '...')
    slice_save(scanID)
print("...Done")
