#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:38:09 2021

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

from true_objective import true_loss
from true_objective import estimated_loss



class routines:
    
    def __init__(self,opt_name,l_rate,epoch_no,dual_pns,pos,neg,d_of_data,naive_initialization=True,init_point=0,betas=(0.9,0.999),alpha=0.1):
        self.optimizer=opt_name
        self.l_rate=l_rate
        self.epoch_no=epoch_no
        self.dual_pns=dual_pns
        self.pos=pos
        self.neg=neg
        self.d_of_data=d_of_data
        self.naive_initialization=naive_initialization
        self.init_point=init_point
        self.betas=betas
        self.alpha=alpha
    
    def run(self):
        if self.optimizer=="SGD_c_l_rate":
            from optimizers import SGD
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD([s], self.l_rate)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/10.0
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses1.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD([s], self.l_rate)
            losses2 = []
            cumul_loss2=[]
            sign=-1
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss2.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            s_tmp=s.clone()
            s_trajectory2=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/10.0
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses2.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss2.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory2=np.append(s_trajectory2,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            return(s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """
        if self.optimizer=="SGD_scale_by_ep":
            from optimizers import SGD
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD([s], self.l_rate)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            for e in range(self.epoch_no):
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses1.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD([s], self.l_rate)
            losses2 = []
            cumul_loss2=[]
            sign=-1
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss2.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            s_tmp=s.clone()
            s_trajectory2=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            for e in range(self.epoch_no):
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses2.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss2.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory2=np.append(s_trajectory2,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            return(s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """
        if self.optimizer=="SGD_adaptive":
            from optimizers import SGD_adaptive
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD_adaptive([s], self.l_rate)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses1.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD_adaptive([s], self.l_rate)
            losses2 = []
            cumul_loss2=[]
            sign=-1
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss2.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            s_tmp=s.clone()
            s_trajectory2=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses2.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss2.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory2=np.append(s_trajectory2,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            return(s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """
        if self.optimizer=="SGD_with_Momentum":
            from optimizers import SGD_with_Momentum
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD_with_Momentum([s], self.l_rate,self.alpha)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            for e in range(self.epoch_no):
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses1.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD_with_Momentum([s], self.l_rate,self.alpha)
            losses2 = []
            cumul_loss2=[]
            sign=-1
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss2.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            s_tmp=s.clone()
            s_trajectory2=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            for e in range(self.epoch_no):
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses2.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss2.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory2=np.append(s_trajectory2,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            return(s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """
        if self.optimizer=="Adagrad":
            from optimizers import AdaGrad
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = AdaGrad([s], self.l_rate,self.betas)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses1.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = AdaGrad([s], self.l_rate,self.betas)
            losses2 = []
            cumul_loss2=[]
            sign=-1
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss2.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            s_tmp=s.clone()
            s_trajectory2=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses2.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss2.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory2=np.append(s_trajectory2,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            return(s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """
        if self.optimizer=="Adam_sg":
            #from optimizers import AdaGrad
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = torch.optim.Adam([s] , lr=self.l_rate)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses1.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = torch.optim.Adam([s] , lr=self.l_rate)
            losses2 = []
            cumul_loss2=[]
            sign=-1
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss2.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            s_tmp=s.clone()
            s_trajectory2=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        optimizer.zero_grad()
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        optimizer.zero_grad()
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                        losses2.append(loss.item())
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss2.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory2=np.append(s_trajectory2,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            return(s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """                  
        if self.optimizer=="Adam_full_batch":
            #from optimizers import AdaGrad
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = torch.optim.Adam([s] , lr=self.l_rate)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            for e in range(self.epoch_no):
                optimizer.zero_grad()
                loss=true_loss(self.dual_pns,self.pos,self.neg,s)
                tmp_loss=loss.item()
                loss.backward()
                optimizer.step()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()

            return(s_trajectory1,cumul_loss1,0,0,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """
        if self.optimizer=="GD_c_l_rate":
            from optimizers import SGD
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD([s], self.l_rate)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            for e in range(self.epoch_no):
                optimizer.zero_grad()
                loss=true_loss(self.dual_pns,self.pos,self.neg,s)
                tmp_loss=loss.item()
                loss.backward()
                optimizer.step()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()

            return(s_trajectory1,cumul_loss1,0,0,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """
        if self.optimizer=="GD_scale_by_ep":
            from optimizers import SGD
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
            optimizer = SGD([s], self.l_rate)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            for e in range(self.epoch_no):
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                optimizer.zero_grad()
                loss=true_loss(self.dual_pns,self.pos,self.neg,s)
                tmp_loss=loss.item()
                loss.backward()
                optimizer.step()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()

            return(s_trajectory1,cumul_loss1,0,0,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """
        if self.optimizer=="SVRG":
            #from optimizers import AdaGrad
            from optimizers import SVRG
            #epoch_no=10
            #l_rate=0.001
            # we have two optimization problems: Optimize both of them. Select optimal s
            #yielding the minimum loss value. So, we will have two copies of variables.
            s = torch.zeros(self.d_of_data, requires_grad=True)
            check_point_s = torch.zeros(self.d_of_data, requires_grad=True)

            group = [{'params': [s], 'lr': self.l_rate, 'checkpoints': [check_point_s]}]
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
                check_point_s=torch.tensor(initialize, requires_grad=True)
                group = [{'params': [s], 'lr': self.l_rate, 'checkpoints': [check_point_s]}]
            optimizer = SVRG(group)
            losses1 = []
            cumul_loss1=[]
            s_tmp=s.clone()
            s_trajectory1=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            
            sign=1
            tmp_min=inf
            
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss1.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        loss.backward()
                        losses1.append(loss.item())
                        
                        
                        checkpoint_loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), check_point_s,sign)
                        checkpoint_loss.backward()
                        
                        
                        optimizer.step()
                
                optimizer.switch_checkpoint_phase()        
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss1.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory1=np.append(s_trajectory1,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            
            
            
            
            
            
            
            s = torch.zeros(self.d_of_data, requires_grad=True)
            check_point_s = torch.zeros(self.d_of_data, requires_grad=True)

            group = [{'params': [s], 'lr': self.l_rate, 'checkpoints': [check_point_s]}]
            if self.naive_initialization==False:
                initialize=self.init_point
                s=torch.tensor(initialize, requires_grad=True)
                check_point_s=torch.tensor(initialize, requires_grad=True)
                group = [{'params': [s], 'lr': self.l_rate, 'checkpoints': [check_point_s]}]
            optimizer = SVRG(group)            
          
            losses2 = []
            cumul_loss2=[]
            sign=-1
            
            tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
            cumul_loss2.append(tmp_loss)
            
            if tmp_loss<tmp_min:
                tmp_min=tmp_loss
                s_tmp=s.clone()
                best_s=s_tmp.detach().numpy()
            
            
            s_tmp=s.clone()
            s_trajectory2=s_tmp.detach().numpy().reshape(1,self.d_of_data)
            for e in range(self.epoch_no):
                #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/(e+1)
                for p in range(len(self.pos)):
                    pos_samp=self.pos[p,:]
                    for n in range(len(self.neg)):
                        
                        neg_samp=self.neg[n,:]
                        t=p*len(self.neg) + n
                        
                        
                        
                        loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), s,sign)
                        loss.backward()
                        losses2.append(loss.item())
                        
                        checkpoint_loss = estimated_loss(torch.from_numpy(np.array([self.dual_pns[t]])), torch.from_numpy(pos_samp), torch.from_numpy(neg_samp), check_point_s,sign)
                        checkpoint_loss.backward()
                        
                        
                        optimizer.step()
                        
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                optimizer.switch_checkpoint_phase()        
                #ax1.scatter(s[0].item(),s[1].item(),c='yellow')
                tmp_loss=true_loss(self.dual_pns,self.pos,self.neg,s).item()
                cumul_loss2.append(tmp_loss)
                s_tmp=s.clone()
                s_trajectory2=np.append(s_trajectory2,s_tmp.detach().numpy().reshape(1,self.d_of_data),axis=0)
           
                if tmp_loss<tmp_min:
                    #print(e)
                    tmp_min=tmp_loss
                    s_tmp=s.clone()
                    best_s=s_tmp.detach().numpy()
            
            return(s_trajectory1,cumul_loss1,s_trajectory2,cumul_loss2,best_s,tmp_min)
            """end
            of 
            SGD_c_l_rate
            you
            can
            add 
            new
            routine below.
            """                                                               
        
        
    