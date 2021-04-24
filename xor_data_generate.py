#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:22:19 2019

@author: can
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import math

def produce_xor_and_candidates():
    np.random.seed(4)
    random.seed(8)
    mu,sigma=0,0.2
    no_point_in_each_region=20
    how_close=1
    max_no_outlier=4
    
    
    #region1
    tmp=np.random.normal(mu,sigma,no_point_in_each_region*2)
    tmp=np.round(tmp,2)
    tmp=tmp.reshape(no_point_in_each_region,2)
    r1=tmp+[0,0]
    w_class_r1 = np.ones((no_point_in_each_region,3))
    w_class_r1[:,:-1] = r1
    
    no_outlier=random.randint(1,max_no_outlier)
    tmp=np.random.normal(mu,sigma,no_outlier*2)
    tmp=np.round(tmp,2)
    tmp=tmp.reshape(no_outlier,2)
    r1_out=tmp+[0,0]
    w_class_r1_out = np.ones((no_outlier,3))*2
    w_class_r1_out[:,:-1] = r1_out
    
    r1=np.append(r1,r1_out,axis=0)
    w_class_r1=np.append(w_class_r1,w_class_r1_out,axis=0)
    
    
    #region2
    tmp=np.random.normal(mu,sigma,no_point_in_each_region*2)
    tmp=np.round(tmp,2)
    tmp=tmp.reshape(no_point_in_each_region,2)
    r2=tmp+[how_close,0]
    w_class_r2 = np.ones((no_point_in_each_region,3))*2
    w_class_r2[:,:-1] = r2
    
    no_outlier=random.randint(1,max_no_outlier)
    tmp=np.random.normal(mu,sigma,no_outlier*2)
    tmp=np.round(tmp,2)
    tmp=tmp.reshape(no_outlier,2)
    r2_out=tmp+[how_close,0]
    w_class_r2_out = np.ones((no_outlier,3))
    w_class_r2_out[:,:-1] = r2_out
    
    r2=np.append(r2,r2_out,axis=0)
    w_class_r2=np.append(w_class_r2,w_class_r2_out,axis=0)
    
    #region3
    tmp=np.random.normal(mu,sigma,no_point_in_each_region*2)
    tmp=np.round(tmp,2)
    tmp=tmp.reshape(no_point_in_each_region,2)
    r3=tmp+[0,how_close]
    w_class_r3 = np.ones((no_point_in_each_region,3))*2
    w_class_r3[:,:-1] = r3
    
    no_outlier=random.randint(1,max_no_outlier)
    tmp=np.random.normal(mu,sigma,no_outlier*2)
    tmp=np.round(tmp,2)
    tmp=tmp.reshape(no_outlier,2)
    r3_out=tmp+[0,how_close]
    w_class_r3_out = np.ones((no_outlier,3))
    w_class_r3_out[:,:-1] = r3_out
    
    r3=np.append(r3,r3_out,axis=0)
    w_class_r3=np.append(w_class_r3,w_class_r3_out,axis=0)
    
    #region4
    tmp=np.random.normal(mu,sigma,no_point_in_each_region*2)
    tmp=np.round(tmp,2)
    tmp=tmp.reshape(no_point_in_each_region,2)
    r4=tmp+[how_close,how_close]
    w_class_r4 = np.ones((no_point_in_each_region,3))
    w_class_r4[:,:-1] = r4
    
    no_outlier=random.randint(1,max_no_outlier)
    tmp=np.random.normal(mu,sigma,no_outlier*2)
    tmp=np.round(tmp,2)
    tmp=tmp.reshape(no_outlier,2)
    r4_out=tmp+[how_close,how_close]
    w_class_r4_out = np.ones((no_outlier,3))*2
    w_class_r4_out[:,:-1] = r4_out
    
    r4=np.append(r4,r4_out,axis=0)
    w_class_r4=np.append(w_class_r4,w_class_r4_out,axis=0)
    
    
    #df=np.append(r1,r2,axis=0)
    #df=np.append(df,r3,axis=0)
    #df=np.append(df,r4,axis=0)
    
    df_class=np.append(w_class_r1,w_class_r2,axis=0)
    df_class=np.append(df_class,w_class_r3,axis=0)
    df_class=np.append(df_class,w_class_r4,axis=0)
    
    
    
    
    min_x=min(df_class[:,0])
    max_x=max(df_class[:,0])
    min_y=min(df_class[:,1])
    max_y=max(df_class[:,1])
    
    
    
    x_range=np.arange(min_x-0.01,max_x+0.02,0.02)
    y_range=np.arange(min_y-0.01,max_y+0.02,0.02)
    from itertools import product
    candidates=np.array(list(product(x_range, y_range)))
    candidates=np.round(candidates,2)
    
    return candidates,df_class

def plot_xor(df_class):
    f, ax = plt.subplots(1)
    ax.scatter(x=df_class[:,0],y=df_class[:,1],c=df_class[:,2],marker='o')
    #f.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    f.suptitle('XOR', fontsize=12)   
    return(f,ax)
    
def generate_dual_pns(df_class,dist,seed):
    np.random.seed(seed)
    df_class[df_class[:,2]==2,2]=-1

    num_pairs=sum(df_class[:,2]==1)*sum(df_class[:,2]==-1)
    pos=df_class[df_class[:,2]==1,:2]
    neg=df_class[df_class[:,2]==-1,:2]
    d_of_data=len(pos[0,:])
    
    if dist=="uniform":
        dual_pns=np.random.uniform(0,1,num_pairs)
        return(df_class,pos,neg,d_of_data,num_pairs,dual_pns)
    
   
