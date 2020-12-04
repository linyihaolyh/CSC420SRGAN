# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:38:52 2020

@author: linyi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self,csize=3):
        super(SRCNN,self).__init__()
        self.lrelu=F.LeakyRelu()
        self.conv1=nn.Conv2d(csize=3,64,kernal_size=9,padding=9//2)
        self.conv2=nn.Conv2d(64,32,kernal_size=7,padding=7//2)
        self.conv3=nn.Conv2d(32,32,kernal_size=5,padding=5//2)
        self.conv4=nn.Conv2d(32,csize,kernal_size=5,padding=5//2)
        
        
        
    def forward(self,x):
        x=self.lrelu(self.conv1(x))
        x=self.lrelu(self.conv2(x))
        x=self.lrelu(self.conv3(x))
        x=self.conv4(x)
        
        return x