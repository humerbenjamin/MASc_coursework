#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
   Problem Chapter 3: OI
   Based on Daley's Fig. 4.7

   Author: Saroja Polavarapu Dec. 31, 2003
   Adapted from Matlab to Python by Tyler W. 2019
------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
x1 = -2.0		    # position of obs 1
x0 = 0.0				# position of analysis point
dx = 0.1            # spacing of x values on grid
x = np.arange(-4,4+dx,dx)	# x-grid
x2 = x					# position of obs 2
alfa = 0.25	    # alfa = var(obs err)/var(bkgd err)

# Preliminary calculations
alfp1 = 1.0+alfa
alfp1sq = alfp1**2.
d10 = abs(x1-x0)	   # normalized distance between points 1 and 0
d12 = abs(x1-x2)   # normalized distance between points 1 and 2
d20 = abs(x2-x0)	   # normalized distance between points 2 and 0
rho10 = np.dot((1+d10),np.exp(-d10))

# Calculate weights for obs 1 and obs 2
rho20 = (1.0+d20) * np.exp(-d20)
rho12 =(1.0+d12) * np.exp(-d12)
w1a = np.dot(rho10,alfp1) - (rho12*rho20)
w2a = np.dot(rho20,alfp1) - np.dot(rho12,rho10)
denom = alfp1sq - (rho12 * rho12)
w1 = w1a/denom
w2 = w2a/denom

# Now calculate the analysis error variance
# normalized by the background error variance
tmp = (np.dot(alfp1,(rho10**2 + (rho20*rho20))) - 2*np.dot(rho10,rho20)*rho12)/denom
epssq = 1-tmp # Analysis error variance 

# Plot results
fig = plt.figure(figsize=[10,10])
plt.title(r'$x_1$ = {}, $\alpha$ = {}'.format(x1,alfa))
plt.plot(x,w1,label='w1')
plt.plot(x,w2,label='w2')
plt.plot(x,epssq,label='Analysis error variance')
plt.xlabel('x/L')
plt.ylim(-0.2,1.2)
plt.xlim(-4,4)
plt.tick_params(axis='both', which='both', labelleft='on',right='on',top='on',direction='in')
plt.legend(loc='upper right') # Add legend

# Add horizontal and vertical lines to the plot
plt.axhline(0,color='k',ls=':')
plt.axvline(-2,color='k',ls='--',linewidth=1.0)
plt.axvline(0,color='k',ls='--',linewidth=1.0)

plt.show()
