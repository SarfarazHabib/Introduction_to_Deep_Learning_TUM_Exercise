import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from torchvision import transforms, utils

# tranforms


class Normalize(object):
    """Normalizes keypoints.
    """
    def __call__(self, sample):
        
        image, key_pts = sample['image'], sample['keypoints']
        
        ##############################################################
        # TODO: Implemnet the Normalize function, where we normalize #
        # the image from [0, 255] to [0,1] and keypoints from [0, 96]#
        # to [-1, 1]                                                 #
        ##############################################################

        image_copy = np.copy(image)
        #print("here is debugging")
        #print(image_copy.min())
        #print(image_copy.max())
        key_pts_copy = np.copy(key_pts)
        #print(key_pts_copy.min())
        #print(key_pts_copy.max())
        #print(key_pts_copy.mean())
        #print(np.sqrt(key_pts_copy))

        # convert image to grayscale
        #image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        #print(image_copy.min())
        #print(image_copy.max())    
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        #key_pts_copy = (key_pts_copy - (key_pts_copy.mean()))/(np.sqrt(key_pts_copy))
        #norm = transforms.Normalize(-1, 1)
        #norm(key_pts_copy)
        key_pts_copy = 2*(key_pts_copy - np.min(key_pts_copy))/np.ptp(key_pts_copy)-1
        #key_pts_copy=  (key_pts_copy - np.mean(key_pts_copy)/np.std(key_pts_copy))
        #print(key_pts_copy.min())
        #print(key_pts_copy.max())
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return {'image': image_copy, 'keypoints': key_pts_copy}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(key_pts).float()}
