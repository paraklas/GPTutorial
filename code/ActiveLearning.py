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

def Denormalize(X, X_m, X_s):    
    return X_s*X + X_m


if __name__ == "__main__":    
    
    N = 8
    D = 1
    lb = -0.5*np.ones(D)
    ub = 1.0*np.ones(D)
    noise = 0.00
    tol = 1e-4
    nsteps = 20
    
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
    
    plt.figure(1, facecolor = 'w')
    
    for i in range(0,nsteps):
        # Train 
        model.train()
        
        # Predict
        y_pred, y_var = model.predict(X_star)
        y_var = np.abs(np.diag(y_var))[:,None]
        
        # Sample where posterior variance is maximized
        new_X = X_star[np.argmax(y_var),:]   
        
        # Check for convergence
        if np.max(y_var) < tol:
            print("Converged!")
            break
        
        # Normalize new point if needed
        if Normalize_input_data == 1:
            xx = Denormalize(new_X, X_m, X_s)
            new_y = f(xx) + + noise*np.random.randn(1,D)
        else:
            new_y = f(new_X) + + noise*np.random.randn(1,D)
        if Normalize_output_data == 1:
            new_y = Normalize(new_y, y_m, y_s)
           
        # Plot
        plt.cla()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)
        plt.plot(X_star, y_star, 'b-', label = "Exact", linewidth=2)
        plt.plot(X_star, y_pred, 'r--', label = "Prediction", linewidth=2)
        lower = y_pred - 2.0*np.sqrt(y_var)
        upper = y_pred + 2.0*np.sqrt(y_var)
        plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                         facecolor='orange', alpha=0.5, label="Two std band")
        plt.plot(model.X,model.y,'bo', label = "Data")
        plt.plot(new_X*np.ones(2), np.linspace(-4,4,2),'k--')
        ax = plt.gca()
        ax.set_xlim([lb[0], ub[0]])
        ax.set_ylim([-4,4])
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title("Iteration #%d" % (i+1))
        plt.pause(0.5)
        plt.savefig("../figures/AL_it_%d.png" % (i+1), format='png', dpi=300)
        
        # Add new point to the training set
        model.X = np.vstack((model.X, new_X))
        model.y = np.vstack((model.y, new_y))
    



   
   