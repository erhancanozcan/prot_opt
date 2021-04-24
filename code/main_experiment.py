#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:42:45 2021

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
import pandas as pd


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
from true_objective import true_loss
from true_objective import estimated_loss
from objective_plot import contour_plot 
from available_routines import routines

#%%
dist_names=['uniform']
seed_list=np.arange(20)

optimizers=["SGD_c_l_rate","SGD_scale_by_ep","SGD_adaptive","SGD_with_Momentum",\
            "Adagrad","Adam_sg","Adam_full_batch","GD_c_l_rate","GD_scale_by_ep","SVRG"]
    
    

naive_initialization=[True,False]

learning_rate=[0.0001,0.001,0.01,0.1,1.0]

epoch=[20,50,100]

momentum_list=[0.1,0.2]


df = pd.DataFrame(columns=['Dual_dist','seed','Algorithm','naive_init','lr','epoch_num'\
                           ,'momentum','best_obj','max_grid_obj','max_data_obj'])



for dist in dist_names:
    for seed in seed_list:
        #genereta dual coefficients:
        df_class,pos,neg,d_of_data,num_pairs,dual_pns=generate_dual_pns(df_class,dist,seed)
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
        for opt in optimizers:
            for nv_init in naive_initialization:
                for lr in learning_rate:
                    for ep in epoch:
                        if opt=="SGD_with_Momentum":
                            for momentum in momentum_list:
                                routine1=routines(opt,l_rate=lr,epoch_no=ep,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=nv_init,init_point=maximizer_data_point,betas=(0.9,0.999),alpha=momentum)
                                s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()
                                tmp={'Dual_dist':dist,'seed':seed,'Algorithm':opt,\
                                     'naive_init':nv_init,'lr':lr,'epoch_num':ep\
                                      ,'momentum':momentum,'best_obj':tmp_min,\
                                     'max_grid_obj':grid_point_obj,'max_data_obj':data_point_obj}

                                df = df.append(tmp, ignore_index = True)
                        else:
                            routine1=routines(opt,l_rate=lr,epoch_no=ep,dual_pns=dual_pns,pos=pos,neg=neg,d_of_data=d_of_data,naive_initialization=nv_init,init_point=maximizer_data_point,betas=(0.9,0.999),alpha=momentum)
                            s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min=routine1.run()
                            tmp={'Dual_dist':dist,'seed':seed,'Algorithm':opt,\
                                     'naive_init':nv_init,'lr':lr,'epoch_num':ep\
                                      ,'momentum':'NA','best_obj':tmp_min,\
                                     'max_grid_obj':grid_point_obj,'max_data_obj':data_point_obj}

                            df = df.append(tmp, ignore_index = True)
                            
                        
                        
                    
                
            
        
        
        
        
        
        