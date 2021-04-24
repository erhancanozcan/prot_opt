#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 23:14:38 2021

@author: can
"""
import torch
from torch.optim import Optimizer
import os
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import random


def true_loss(dual_pns,pos,neg,s):

    loss=torch.zeros(1)
    p=0
    for p in range(len(pos)):
        pos_samp=pos[p,:]
        pos_samp=torch.from_numpy(pos_samp)
        for n in range(len(neg)):
            neg_samp=neg[n,:]
            neg_samp=torch.from_numpy(neg_samp)
            t=p*len(neg) + n
            
            tmp=(torch.norm(pos_samp-s,p=2)) - (torch.norm(neg_samp-s,p=2))
            loss.add_(tmp,alpha=dual_pns[t])
            
    loss=torch.abs(loss) 
    return(-loss)

def estimated_loss(dual_v,pos_samp, neg_samp, s1,sign=1):
    #-1: You want to obtain maximum positive value inside of the absolute value.
    #by multiplying -1, make it minimization problem. Note that objective value of maximization problem
    #is optimal objective*-1.
    
    #1 minimizes the inside of absolute value. Your goal is to get the most negative objective.
    #If true loss is negative at the final step. Probably, it is better to optimize maximization problem.
    loss=sign*(dual_v*(torch.norm(pos_samp-s1,p=2)) - dual_v*(torch.norm(neg_samp-s1,p=2)))
    return loss

def SVRG_true_loss(dual_pns,pos,neg,s,sign=1):

    loss=torch.zeros(1)
    p=0
    for p in range(len(pos)):
        pos_samp=pos[p,:]
        pos_samp=torch.from_numpy(pos_samp)
        for n in range(len(neg)):
            neg_samp=neg[n,:]
            neg_samp=torch.from_numpy(neg_samp)
            t=p*len(neg) + n
            
            tmp=(torch.norm(pos_samp-s,p=2)) - (torch.norm(neg_samp-s,p=2))
            loss.add_(tmp,alpha=dual_pns[t])
            
    loss=sign*loss
    return(loss)



