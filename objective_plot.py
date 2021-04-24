#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 23:50:01 2021

@author: can
"""
import numpy as np


def distance_matrix(A, B, squared=False):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared


def contour_plot(candidates,pos,neg,dual_pns,no_grid=False):
    cand_pos=distance_matrix(candidates, pos, squared=False)
    cand_neg=distance_matrix(candidates, neg, squared=False)
    candidates_loss=[]
    for i in  range(len(candidates)):
        grid_loss=0
        for p in range(len(pos)):
            pos_samp=pos[p,:]
            for n in range(len(neg)):
                neg_samp=neg[n,:]
                t=p*len(neg) + n
                    
                grid_loss+=dual_pns[t]*(cand_pos[i,p] - cand_neg[i,n])
        candidates_loss.append(abs(grid_loss))
    
    if no_grid==True:
        return(candidates_loss)
    
    
    candidates_loss=np.array([candidates_loss])
    candidates_loss=candidates_loss.reshape(len(np.unique(candidates[:,0])),len(np.unique(candidates[:,1])))
    candidates_loss=candidates_loss.T
    X, Y = np.meshgrid(np.unique(candidates[:,0]), np.unique(candidates[:,1]))
    
    return(X,Y,candidates_loss)
    
    
    