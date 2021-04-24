#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:13:02 2021

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
from numpy import inf
#%%

folder_location="/Users/can/Desktop/phd/se 724/papers/code"
os.chdir(folder_location)
from xor_data_generate import produce_xor_and_candidates

candidates,df_class =produce_xor_and_candidates()
#plot raw data.
from xor_data_generate import plot_xor
f,ax= plot_xor(df_class)
#%%

from xor_data_generate import generate_dual_pns
dist="uniform"
seed=4 
df_class,pos,neg,d_of_data,num_pairs,dual_pns=generate_dual_pns(df_class,dist,seed)
#%%

from true_objective import true_loss
from true_objective import estimated_loss
from true_objective import SVRG_true_loss

#contour plot
from objective_plot import contour_plot 
X,Y,Z=contour_plot(candidates,pos,neg,dual_pns)

#different initialization strategies: maximizier_data_point, maximizer_grid_point
Z_data_points=contour_plot(df_class[:,:2],pos,neg,dual_pns,no_grid=True)

maximizer_data_point=df_class[np.argmax(Z_data_points),:2]
maximizer_grid_point=np.zeros(d_of_data,int)
for i in range (len(np.where(Z==np.max(Z)))):
    maximizer_grid_point[i]=int((np.where(Z==np.max(Z)))[i][0])
maximizer_grid_point=np.array([X[maximizer_grid_point[0],maximizer_grid_point[1]],Y[maximizer_grid_point[0],maximizer_grid_point[1]]])

data_point_obj=-np.max(Z_data_points)
grid_point_obj=-np.max(Z)

#%%
naive_initialization=True
initialize=0
if naive_initialization==False:
    initialize=maximizer_data_point
    s=torch.tensor(maximizer_data_point, requires_grad=True)

from available_routines import routines

routine1=routines("SGD_c_l_rate",l_rate=0.001,epoch_no=10,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()

routine1=routines("SGD_scale_by_ep",l_rate=0.0001,epoch_no=50,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()

routine1=routines("SGD_adaptive",l_rate=0.0001,epoch_no=10,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()

routine1=routines("SGD_with_Momentum",l_rate=0.001,epoch_no=10,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()

routine1=routines("Adagrad",l_rate=0.0001,epoch_no=10,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()

routine1=routines("Adam_sg",l_rate=0.001,epoch_no=10,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()

routine1=routines("Adam_full_batch",l_rate=0.1,epoch_no=100,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,_,_,best_s,tmp_min=routine1.run()

routine1=routines("GD_c_l_rate",l_rate=0.001,epoch_no=100,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,_,_,best_s,tmp_min=routine1.run()

routine1=routines("GD_scale_by_ep",l_rate=0.001,epoch_no=100,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,_,_,best_s,tmp_min=routine1.run()

routine1=routines("SVRG",l_rate=0.001,epoch_no=100,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=naive_initialization,init_point=initialize,betas=(0.9,0.999),alpha=0.1)
s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()



#best_s
#Out[171]: array([-0.140496  , -0.15884009], dtype=float32)

#tmp_min
#Out[172]: -151.44775390625
#%%
from optimizers import SVRG
epoch_no=10
lr=0.001
# we have two optimization problems: Optimize both of them. Select optimal s
#yielding the minimum loss value. So, we will have two copies of variables.
s = torch.zeros(d_of_data, requires_grad=True)
check_point_s = torch.zeros(d_of_data, requires_grad=True)

group = [{'params': [s], 'lr': lr, 'checkpoints': [check_point_s]}]

#s_v=s
optimizer = SVRG(group)
losses1 = []
cumul_loss1=[]

s_tmp=s.clone()
s_trajectory1=s_tmp.detach().numpy().reshape(1,d_of_data)

sign=1
tmp_min=inf


tmp_loss=true_loss(dual_pns,pos,neg,s).item()
cumul_loss1.append(tmp_loss)

if tmp_loss<tmp_min:
    tmp_min=tmp_loss
    s_tmp=s.clone()
    best_s=s_tmp.detach().numpy()

p=0
n=0

for e in range(epoch_no):
    #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/10.0
    for p in range(len(pos)):
        pos_samp=pos[p,:]
        for n in range(len(neg)):
            #optimizer.zero_grad()
            neg_samp=neg[n,:]
            t=p*len(neg) + n
            
            #optimizer.zero_grad()
            
            loss = estimated_loss(torch.from_numpy(np.array([dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
            losses1.append(loss.item())
            loss.backward()
            
            checkpoint_loss=estimated_loss(torch.from_numpy(np.array([dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp),check_point_s,sign)
            checkpoint_loss.backward()
            
            optimizer.step()
    
    optimizer.switch_checkpoint_phase()
    #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
    tmp_loss=true_loss(dual_pns,pos,neg,s).item()
    cumul_loss1.append(tmp_loss)
    s_tmp=s.clone()
    s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,d_of_data),axis=0)
    if tmp_loss<tmp_min:
        #print(e)
        tmp_min=tmp_loss
        best_s=s.detach().numpy()
        
s = torch.zeros(d_of_data, requires_grad=True)
check_point_s = torch.zeros(d_of_data, requires_grad=True)

group = [{'params': [s], 'lr': lr, 'checkpoints': [check_point_s]}]

optimizer = SVRG(group)
losses2 = []
cumul_loss2=[]

s_tmp=s.clone()
s_trajectory2=s_tmp.detach().numpy().reshape(1,d_of_data)

sign=-1


tmp_loss=true_loss(dual_pns,pos,neg,s).item()
cumul_loss2.append(tmp_loss)

if tmp_loss<tmp_min:
    tmp_min=tmp_loss
    s_tmp=s.clone()
    best_s=s_tmp.detach().numpy()
    

for e in range(epoch_no):
    #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/10.0
    for p in range(len(pos)):
        pos_samp=pos[p,:]
        for n in range(len(neg)):
            #optimizer.zero_grad()
            neg_samp=neg[n,:]
            t=p*len(neg) + n
            
            #optimizer.zero_grad()
            
            loss = estimated_loss(torch.from_numpy(np.array([dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
            losses2.append(loss.item())
            loss.backward()
            
            checkpoint_loss=estimated_loss(torch.from_numpy(np.array([dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp),check_point_s,sign)
            checkpoint_loss.backward()
            
            optimizer.step()
    
    optimizer.switch_checkpoint_phase()
    #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
    tmp_loss=true_loss(dual_pns,pos,neg,s).item()
    cumul_loss2.append(tmp_loss)
    s_tmp=s.clone()
    s_trajectory2=np.append(s_trajectory2,s_tmp.detach().numpy().reshape(1,d_of_data),axis=0)
    if tmp_loss<tmp_min:
        #print(e)
        tmp_min=tmp_loss
        s_tmp=s.clone()
        best_s=s_tmp.detach().numpy()

#%%
from optimizers import SGD
epoch_no=10
l_rate=0.0001
# we have two optimization problems: Optimize both of them. Select optimal s
#yielding the minimum loss value. So, we will have two copies of variables.
#s = torch.zeros(d_of_data, requires_grad=True)
s=torch.tensor(maximizer_data_point, requires_grad=True)
optimizer = SGD([s], l_rate)
losses1 = []
cumul_loss1=[]
s_tmp=s.clone()
s_trajectory1=s_tmp.detach().numpy().reshape(1,d_of_data)

sign=1
tmp_min=inf


tmp_loss=true_loss(dual_pns,pos,neg,s).item()
cumul_loss1.append(tmp_loss)

if tmp_loss<tmp_min:
    tmp_min=tmp_loss
    s_tmp=s.clone()
    best_s=s_tmp.detach().numpy()


for e in range(epoch_no):
    #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/10.0
    for p in range(len(pos)):
        #print(s_trajectory1)
        pos_samp=pos[p,:]
        for n in range(len(neg)):
            optimizer.zero_grad()
            neg_samp=neg[n,:]
            t=p*len(neg) + n
            
            optimizer.zero_grad()
            
            loss = estimated_loss(torch.from_numpy(np.array([dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
            
            loss.backward()
            
            optimizer.step()
    
            losses1.append(loss.item())
    #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
    tmp_loss=true_loss(dual_pns,pos,neg,s).item()
    cumul_loss1.append(tmp_loss)
    s_tmp=s.clone()
    s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,d_of_data),axis=0)
    if tmp_loss<tmp_min:
        #print(e)
        tmp_min=tmp_loss
        s_tmp=s.clone()
        best_s=s_tmp.detach().numpy()

#s = torch.zeros(d_of_data, requires_grad=True)
s=torch.tensor(maximizer_data_point, requires_grad=True)
optimizer = SGD([s], l_rate)
losses2 = []
cumul_loss2=[]
sign=-1

tmp_loss=true_loss(dual_pns,pos,neg,s).item()
cumul_loss2.append(tmp_loss)

if tmp_loss<tmp_min:
    tmp_min=tmp_loss
    s_tmp=s.clone()
    best_s=s_tmp.detach().numpy()


s_tmp=s.clone()
s_trajectory2=s_tmp.detach().numpy().reshape(1,d_of_data)
for e in range(epoch_no):
    #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/10.0
    for p in range(len(pos)):
        pos_samp=pos[p,:]
        for n in range(len(neg)):
            optimizer.zero_grad()
            neg_samp=neg[n,:]
            t=p*len(neg) + n
            
            optimizer.zero_grad()
            
            loss = estimated_loss(torch.from_numpy(np.array([dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
            
            loss.backward()
            
            optimizer.step()
    
            losses2.append(loss.item())
    #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
    tmp_loss=true_loss(dual_pns,pos,neg,s).item()
    cumul_loss2.append(tmp_loss)
    s_tmp=s.clone()
    s_trajectory2=s_tmp.detach().numpy().reshape(1,d_of_data)
    if tmp_loss<tmp_min:
        #print(e)
        tmp_min=tmp_loss
        s_tmp=s.clone()
        best_s=s_tmp.detach().numpy()
        
       
#%%






#%%
#ax1.scatter(s[0].item(),s[1].item(),c='blue',label="GD")
for i in range(20):
    optimizer.zero_grad()
    #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/2
    loss=true_loss(dual_pns,pos,neg,s)
    loss.backward()
    optimizer.step()
    #ax1.scatter(s[0].item(),s[1].item(),c='blue')
    print(loss.item())
#%%



