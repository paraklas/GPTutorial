#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:02:36 2017

@author: Paris
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from gaussian_process import GP

np.random.seed(1234)

def f(x):
    return x * np.sin(4.0*np.pi*x)

def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)


if __name__ == "__main__":    
    
    N = 12
    D = 1
    lb = -0.5*np.ones(D)
    ub = 1.0*np.ones(D)
    noise = 0.00
    
    Normalize_input_data = 1
    Normalize_output_data = 1
    
    # Training data    
    X = lb + (ub-lb)*lhs(D, N)
    y = f(X) + noise*np.random.randn(N,D)
    
    # Test data
    nn = 200
    X_star = np.linspace(lb, ub, nn)[:,None]
    y_star = f(X_star)
    
     #  Normalize Input Data
    if Normalize_input_data == 1:
        X_m = np.mean(X, axis = 0)
        X_s = np.std(X, axis = 0)   
        X = Normalize(X, X_m, X_s)
        lb = Normalize(lb, X_m, X_s)
        ub = Normalize(ub, X_m, X_s)
        X_star = Normalize(X_star, X_m, X_s)
        
    #  Normalize Output Data
    if Normalize_output_data == 1:
        y_m = np.mean(y, axis = 0)
        y_s = np.std(y, axis = 0)   
        y = Normalize(y, y_m, y_s)
        y_star = Normalize(y_star, y_m, y_s)
    
    # Define model
    model = GP(X, y)
    
    # Train 
    model.train()
    
    # Predict
    y_pred, y_var = model.predict(X_star)
    y_var = np.abs(np.diag(y_var))
           
    # Check accuracy                  
    error = np.linalg.norm(y_pred-y_star,2)/np.linalg.norm(y_star,2)
    print("Relative L2 error u: %e" % (error))
    
    # Draw samples from the prior and posterior
    Y0 = model.draw_prior_samples(X_star, 100)
    YP = model.draw_posterior_samples(X_star, 100)
    
    # Plot predictions
    plt.figure(1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.plot(X_star, y_star, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, y_pred, 'r--', label = "Prediction", linewidth=2)
    lower = y_pred - 2.0*np.sqrt(y_var[:,None])
    upper = y_pred + 2.0*np.sqrt(y_var[:,None])
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X,y,'bo', markersize = 12, alpha = 0.5, label = "Data")
    plt.legend(frameon=False,loc='upper left')
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    
    # Plot samples
    plt.figure(2, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.plot(X_star,Y0)
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.ylabel('$f(x)$')
    plt.title("Prior samples")
    plt.subplot(2,1,2)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.plot(X_star,YP)
    plt.plot(X,y,'bo', markersize = 12, alpha = 0.5, label = "Data")
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title("Posterior samples")
    

    
   
   