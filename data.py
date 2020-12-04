# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:16:49 2020

@author: linyi
"""
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np



CROP_SIZE=32




def center_crop(img, new_width, new_height):        

    width = img.shape[1]
    height = img.shape[0]


    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


class DataFromFolder(torch.utils.data.Dataset):
    def __init__(self,filepath,scale_factor):
        super(DataFromFolder, self).__init__()
        self.scale_factor=scale_factor
        self.filenames=[]
        for file in listdir(filepath):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                self.filenames.append(join(filepath,file))
        
    def __getitem__(self,index):
        input_=Image.open(self.filenames[index]).convert('RGB')
        input_=center_crop(input_, CROP_SIZE, CROPSIZE)
        target=input_.copy()
        
        input_=input_.resize(target.size[0]//scale_factor,target.size[1]//scale_factor)
        input_=input_resize(target_img.size,Image.BICUBIC)
        
        
      
        return transforms.ToTensor()(input_), transforms.ToTensor()(target)
        
        
        
    def __len__(self):
        return len(self.filenames)