#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:33:24 2021

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



class SGD_adaptive(Optimizer):
  def __init__(self, params, lr=1.0):
    super(SGD_adaptive, self).__init__(params, {'lr': lr})

    for group in self.param_groups:
      for param in group['params']:
        state = self.state[param]
        state['step'] = 0

        # make a dictionary entry for Gt. It is just a 1-D vector (a scalar).
        # device=p.device tells pytorch to allocate the memory for Gt on the 
        # same device (e.g. CPU or GPU) as the data for the variable p.
        state['Gt'] = torch.zeros(1, device=param.device)

  @torch.no_grad()
  def step(self, closure=None):
    epsilon=1e-8
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr = group['lr']
      for param in group['params']:
        if param.grad is None:
          continue

        # Update the iteration counter (again, this is not actually used in this algorithm)
        state = self.state[param]
        state['step'] += 1
        step = state['step']

        grad = param.grad

        state['Gt'].add_(torch.norm(grad)**2)
        
        param.addcdiv_(grad, torch.sqrt(state['Gt']).add_(epsilon), value=-lr)



class SGD(Optimizer):
  def __init__(self, params, lr=1.0):
    super(SGD, self).__init__(params, {'lr': lr})

    # The params argument can be a list of pytorch variables, or
    # a list of dicts. If it is a list of dicts, each dict should have 
    # a key 'params' that is a list of pytorch variables,
    # and optionally another key 'lr' that specifies the learning rate
    # for those variables. If 'lr' is not provided, the default value
    # is the single value provided as an argument after params to this
    # constructor.
    # If params is just a list of pytorch variables, it is the same
    # as if params were actually a list containing a single dictionary
    # whose 'params' key value is the list of variables.
    # See examples in following code blocks for use of params.

    # Set up an iteration counter.
    # self.state[p] is a python dict for each parameter p
    # that can be used to store various state useful in the optimization
    # algorithm. In this case, we simply store the iteration count, although
    # it is not used in this simple algorithm.
    for group in self.param_groups:
      for p in group['params']: ## same as for p in params: usually.
        # in all code you need to write for this class, len(self.param_groups)==1,
        # and group['params']==params, the argument to the "__init__" function.
        # but this might not hold more generally.
        state = self.state[p]
        state['step'] = 0
        


  @torch.no_grad()
  def step(self, closure=None):
    # in this class, and also usually in practice, closure will always be None.
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      # this lr value is set up by the base class constructor if the 'params'
      # argument to __init__ is a list.
      lr = group['lr']

      # it is common practice to call the model parameters p in code.
      # in class we follow more closely analytical conventions, in which the
      # parameters are often called w for weights.
      for p in group['params']:
        if p.grad is None:
          continue
        
        # Update the iteration counter (again, this is not actually used in this algorithm)
        state = self.state[p]
        state['step'] += 1


        
        # Perform the SGD update. p.grad holds the gradient of the loss
        # with respect to p.
        # Morally, we want to do 
        # p = p - lr * p.grad
        # but because p is actually a pointer, we want to modify
        # the data it points to in-place, which the above will not do
        # (it will allocate new data with the value p - lr*p.grad and then set
        # p as a pointer to that new data).
        # so, instead we use the .add_ function. It has the semantics
        # p.add_(X,y) will change the value pointed to by p by adding c*X to it.
        p.add_(p.grad, alpha=-lr)

class SVRG_tmp(Optimizer):
  def __init__(self, params, lr=1.0):
    super(SVRG_tmp, self).__init__(params, {'lr': lr})

    # The params argument can be a list of pytorch variables, or
    # a list of dicts. If it is a list of dicts, each dict should have 
    # a key 'params' that is a list of pytorch variables,
    # and optionally another key 'lr' that specifies the learning rate
    # for those variables. If 'lr' is not provided, the default value
    # is the single value provided as an argument after params to this
    # constructor.
    # If params is just a list of pytorch variables, it is the same
    # as if params were actually a list containing a single dictionary
    # whose 'params' key value is the list of variables.
    # See examples in following code blocks for use of params.

    # Set up an iteration counter.
    # self.state[p] is a python dict for each parameter p
    # that can be used to store various state useful in the optimization
    # algorithm. In this case, we simply store the iteration count, although
    # it is not used in this simple algorithm.
    for group in self.param_groups:
      for p in group['params']: ## same as for p in params: usually.
        # in all code you need to write for this class, len(self.param_groups)==1,
        # and group['params']==params, the argument to the "__init__" function.
        # but this might not hold more generally.
        state = self.state[p]
        state['step'] = 0
        


  @torch.no_grad()
  def step(self, grad_ell_v, grad_L_v ,closure=None):
    # in this class, and also usually in practice, closure will always be None.
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      # this lr value is set up by the base class constructor if the 'params'
      # argument to __init__ is a list.
      lr = group['lr']

      # it is common practice to call the model parameters p in code.
      # in class we follow more closely analytical conventions, in which the
      # parameters are often called w for weights.
      for p in group['params']:
        if p.grad is None:
          continue
        
        # Update the iteration counter (again, this is not actually used in this algorithm)
        state = self.state[p]
        state['step'] += 1


        
        # Perform the SGD update. p.grad holds the gradient of the loss
        # with respect to p.
        # Morally, we want to do 
        # p = p - lr * p.grad
        # but because p is actually a pointer, we want to modify
        # the data it points to in-place, which the above will not do
        # (it will allocate new data with the value p - lr*p.grad and then set
        # p as a pointer to that new data).
        # so, instead we use the .add_ function. It has the semantics
        # p.add_(X,y) will change the value pointed to by p by adding c*X to it.
        print(p.grad)
        p.add_(p.grad-grad_ell_v+grad_L_v, alpha=-lr)


class SGD_with_Momentum(Optimizer):
  def __init__(self, params, lr=1.0,alpha_rate=0.1):
    super(SGD_with_Momentum, self).__init__(params, {'lr': lr,'alpha_rate':alpha_rate})

    # The params argument can be a list of pytorch variables, or
    # a list of dicts. If it is a list of dicts, each dict should have 
    # a key 'params' that is a list of pytorch variables,
    # and optionally another key 'lr' that specifies the learning rate
    # for those variables. If 'lr' is not provided, the default value
    # is the single value provided as an argument after params to this
    # constructor.
    # If params is just a list of pytorch variables, it is the same
    # as if params were actually a list containing a single dictionary
    # whose 'params' key value is the list of variables.
    # See examples in following code blocks for use of params.

    # Set up an iteration counter.
    # self.state[p] is a python dict for each parameter p
    # that can be used to store various state useful in the optimization
    # algorithm. In this case, we simply store the iteration count, although
    # it is not used in this simple algorithm.
    for group in self.param_groups:
      for p in group['params']: ## same as for p in params: usually.
        # in all code you need to write for this class, len(self.param_groups)==1,
        # and group['params']==params, the argument to the "__init__" function.
        # but this might not hold more generally.
        state = self.state[p]
        state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state['step'] = 0
        


  @torch.no_grad()
  def step(self, closure=None):
    # in this class, and also usually in practice, closure will always be None.
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      # this lr value is set up by the base class constructor if the 'params'
      # argument to __init__ is a list.
      lr = group['lr']
      alpha_rate=group['alpha_rate']

      # it is common practice to call the model parameters p in code.
      # in class we follow more closely analytical conventions, in which the
      # parameters are often called w for weights.
      for p in group['params']:
        if p.grad is None:
          continue
        
        # Update the iteration counter (again, this is not actually used in this algorithm)
        state = self.state[p]
        state['step'] += 1
        
        state['momentum'].add_(state['momentum'],alpha=-alpha_rate)
        state['momentum'].add_(p.grad,alpha=alpha_rate)



        
        # Perform the SGD update. p.grad holds the gradient of the loss
        # with respect to p.
        # Morally, we want to do 
        # p = p - lr * p.grad
        # but because p is actually a pointer, we want to modify
        # the data it points to in-place, which the above will not do
        # (it will allocate new data with the value p - lr*p.grad and then set
        # p as a pointer to that new data).
        # so, instead we use the .add_ function. It has the semantics
        # p.add_(X,y) will change the value pointed to by p by adding c*X to it.
        p.add_(state['momentum'], alpha=-lr)
        #print(state['momentum'])
        
class AdaGrad(Optimizer):
  def __init__(self, params, lr=1.0, betas=(0.9,0.999), decouple=False, debias=True):
    # betas are ignored, but we keep them in the function signature so that it is the same as
    # the adam variants.
    super(AdaGrad, self).__init__(params, {'lr': lr, 'beta1': betas[0], 'beta2': betas[1], 'weight_decay': 0.0})


    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = 0
        state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)


  @torch.no_grad()
  def step(self, closure=None):
    # in this class, and also usually in practice, closure will always be None.
    loss = None
    epsilon = 1e-8
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr = group['lr']
      beta1 = group['beta1']
      beta2 = group['beta2']
      weight_decay = group['weight_decay']

      # it is common practice to call the model parameters p in code.
      # in class we follow more closely analytical conventions, in which the
      # parameters are often called w for weights.
      for p in group['params']:
        if p.grad is None:
          continue

        if weight_decay != 0.0:
          p.grad.add_(p, alpha=weight_decay)
        
        # Update the iteration counter (again, this is not actually used in this algorithm)
        state = self.state[p]
        state['step'] += 1


        state['v'].addcmul_(p.grad, p.grad, value=1.0)

        p.addcdiv_(p.grad, torch.sqrt(state['v']).add_(epsilon), value=-lr)


class SVRG(Optimizer):
  def __init__(self, params, lr=1.0):
    try:
      checkpoints = params[0].get("checkpoints", None)
    except:
      raise TypeError("params argument of SVRG is not a dict-like object.")
    
    super(SVRG, self).__init__(params, {'lr': lr, 'checkpoints':None})

    if checkpoints is None:
      raise SyntaxError("params argument of SVRG should have a 'checkpoints' field.")

    # This variable is used to determin the total number of minibatches in one
    # epoch. If the training set has size N and the batches have B examples per
    # batch, then this number will be N/B. However, the optimizer does not know
    # either N or B. So this will simply be a counter that will increment every
    # iteration. When the epoch is finished, the training function will have to
    # tell the optimizer that it is over, at which point the value of this variable
    # will be the correct value.
    self.N_over_batch_size = 0
    # this next variable is used for debugging: it will check if the batch
    # size has changed or if the N_over_batch_size variable is being updated
    # differently in different phases.
    self.prev_N_over_batch_size = None

    # flags indicating which phase we are in.
    # initial_phase=True means we are in the first epoch, in which we run ordinary
    # adaptive SGD without any variance reduction.
    self.initial_phase=True
    # checkpoint_phase indicates whether this epoch is used to compute a
    # checkpoint gradient or to perform SGD with variance-reduced gradient estimates.
    self.checkpoint_phase = False
    self.lr=lr

    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = 0
        state['checkpoint_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state['Gt'] = torch.zeros(1, device=p.device)


  def process_checkpoint_for_SGD_phase(self, param, state):
    # this function should compute any post-processing (if necessary) to
    # leave state['checkpoint_grad'] holding a value equal to the average of
    # all the checkpoint gradients observed over the course of last epoch.
    # arguments:
    # param: parameter value (w)
    # state: value for self.state[param].

    ## YOUR CODE HERE ##
    state['checkpoint_grad']=torch.div(state['checkpoint_grad'],self.N_over_batch_size)
    #state['checkpoint_grad']=state['checkpoint_grad']/float(self.N_over_batch_size)
    
    #Adding line below significantly improves SVRG for lr=10 and lr=100, but
    #we observe fluctuation when lr=1000.
    #state['Gt'] = torch.zeros(1, device=param.device)
    pass

  def checkpoint_update(self, param, ckpt_grad, state):
    # this function should perform some update using the gradient at the checkpoint
    # in order to eventually have state['checkpoint_grad'] contain the average
    # of all the ckpt_grad's encountered over this epoch.
    # Note that at the end of the epoch, process_checkpoint_for_SGD_phase
    # will be called, so that you can do some final processing to compute the
    # average if you need to.
    # arguments:
    # ckpt_grad: gradient at checkpoint (\nabla \ell(v, z_t))
    # param: parameter value (w)
    # state: value for self.state[param].


    ## YOUR CODE HERE ##

    state['checkpoint_grad'].add_(ckpt_grad)

    

    pass


  def vr_update(self, grad, ckpt_grad, param, state,lr):
    # this function should compute the variance-reduced gradient estimate g_t,
    # update the adaptive learning rate, and compute an SGD update.
    # arguments:
    # grad: gradient at param (\nabla \ell(w_t, z_t))
    # ckpt_grad: gradient at checkpoint (\nabla \ell(v, z_t))
    # param: parameter to update (w_t)
    # state: value for self.state[param].

    ## YOUR CODE HERE ##
    epsilon=1e-8
    g_t=grad-ckpt_grad+state['checkpoint_grad']

    state['Gt'].add_(torch.norm(g_t)**2)
    #print(state)
        
    param.addcdiv_(g_t, torch.sqrt(state['Gt']).add_(epsilon), value=-lr)



    pass


  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    epsilon = 1e-8
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    self.N_over_batch_size += 1
    for group in self.param_groups:
      lr = group['lr']

      for param, ckpt_param in zip(group['params'], group['checkpoints']):
        if param.grad is None:
          continue
        
        assert ckpt_param.grad is not None, "checkpoint should have gradients!"
        ckpt_grad = ckpt_param.grad
        grad = param.grad

        state = self.state[param]
        if self.initial_phase:
          state['Gt'].add_(torch.norm(grad)**2)
          param.addcdiv_(grad, torch.sqrt(state['Gt']).add_(epsilon), value=-lr)

        elif self.checkpoint_phase:
          self.checkpoint_update(param, ckpt_grad, state)
        
        else:
          # Make sure that the smoothness properties underyling SVRG hold (you showed that the loss is 1-smooth in Question 1)
          #assert torch.linalg.norm(grad - ckpt_grad) < 1.001*(0.00001+torch.linalg.norm(param - ckpt_param)) #small values added for floating point rounding errors.

          self.vr_update(grad, ckpt_grad, param, state,lr)
          
           
  @torch.no_grad()
  def switch_checkpoint_phase(self):
    """switches checkpoint_phase to opposite value. performs any necessary
    initialization of variables for this phase."""

    # feel free to uncomment these print statement to help debug
    # if self.checkpoint_phase:
    #   print("next epoch will be an SGD epoch")
    # else:
    #   print("next epoch will be a checkpoint epoch")

    self.checkpoint_phase = not self.checkpoint_phase

    # This is just to check for potential errors in counting the number
    # of iterations in an epoch.
    if self.prev_N_over_batch_size is not None:
      assert self.prev_N_over_batch_size == self.N_over_batch_size
    self.prev_N_over_batch_size = self.N_over_batch_size


    if self.checkpoint_phase:
      for group in self.param_groups:
        for param, ckpt_param in zip(group['params'], group['checkpoints']):
          # verify that the parameter has been updated in the SGD phase:
          #assert torch.linalg.norm(param - ckpt_param) > 0.00001

          # copy over the final iterate produced by SGD into the checkpoint values,
          # and zero out the checkpoint gradient in preparation for computing a new
          # checkpoint gradient in the upcoming checkpoint phase.
          state = self.state[param]
          ckpt_param.copy_(param)
          state['checkpoint_grad'].zero_()
    else:
      #feel free to uncomment this print statement to help debug
      # print("n over batch size: ", self.N_over_batch_size)
      for group in self.param_groups:
        for param, ckpt_param in zip(group['params'], group['checkpoints']):
          # verify that no changes to the parameter occured during the checkpoint phase
          #assert torch.linalg.norm(param - ckpt_param) < 0.0001

          state = self.state[param]

          self.process_checkpoint_for_SGD_phase(param, state)


    self.N_over_batch_size = 0
    self.initial_phase=False


  # The following two functions make sure that the gradients are computed
  # properly at the checkpoint parameter values. It is not important to
  # read or understand them. 
  def zero_grad(self, set_to_none: bool = False):
    self.zero_checkpoint_grad(set_to_none)
    super(SVRG, self).zero_grad(set_to_none)

  def zero_checkpoint_grad(self, set_to_none: bool = False):
    """Sets the gradients of all checkpoints to zero."""
    if not hasattr(self, "_zero_grad_profile_name"):
      self._hook_for_profile()
    with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
      for group in self.param_groups:
        for p in group['checkpoints']:
          if p.grad is not None:
            if set_to_none:
              p.grad = None
            else:
              if p.grad.grad_fn is not None:
                p.grad.detach_()
              else:
                p.grad.requires_grad_(False)
              p.grad.zero_()
        
        
        