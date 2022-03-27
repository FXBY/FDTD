# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:21:42 2020

@author: fbar
"""

import numpy as np
import matplotlib.pyplot as plt

# Method for animating plots
from matplotlib.animation import FuncAnimation


#### Simulation parameters 

## Yee grid parameters
# Number of cells 
N = 200

## Time parameters

# Mumber of time steps
T = 400


## Initialising EM fields

# (normalised) Electric field vectors
E_x = np.zeros((N,T))
E_y = np.zeros((N,T))

# (normalised) Magnetic field vectors
H_x = np.zeros((N,T))
H_y = np.zeros((N,T))


## parameters for Gaussian pulse as a source

# time width of pulse 
tau = 5
# time offset for injecting source (for numerical stability -> no large gradients)
t0 = 30
# cell where source is injected
k0 = 50

# Calculate source
g = np.array([ np.exp(-0.5*((t-t0)/tau)**2) for t in range(T)]) 


#### Main FDTD loop

## Main Loop over time
for t in range(T-1):
    
    ## Update B and H fields
    
    # Loop over cells apart from right boundary
    for k in range(N-1):
        
        # Update B-field from induction law
        H_x[k,t+1] = H_x[k,t] + (E_y[k+1,t] - E_y[k,t])/377
        H_y[k,t+1] = H_y[k,t] - (E_x[k+1,t] - E_x[k,t])/377
        
    # TF/SF correction
    H_x[k0-1,t+1] -= np.exp(-0.5*((t-t0)/tau)**2)/377
    H_y[k0-1,t+1] += np.exp(-0.5*((t-t0)/tau)**2)/377
    
    ## Update D and E field    
    
    # Loop over cells apart from left boundary
    for k in range(1,N): 

        # Update D-Field from Ampere-Maxwell
        E_x[k,t+1] = E_x[k,t] - 377*(H_y[k,t+1] - H_y[k-1,t+1])
        E_y[k,t+1] = E_y[k,t] + 377*(H_x[k,t+1] - H_x[k-1,t+1])
    
    # TF/SF correction
    E_x[k0,t+1] += np.exp(-0.5*((t-t0+0.5-(-0.5))/tau)**2)
    E_y[k0,t+1] += np.exp(-0.5*((t-t0+0.5-(-0.5))/tau)**2)
    
    # Update soft source
    #E_x[k0,t+1] += g[t+1]
    #E_y[k0,t+1] += g[t+1]
    
# Setting up the plotting figure
fig, (ax1, ax2) = plt.subplots(2,1,figsize = (10,6))

# frames per second
fps = 25

# Saving the line2D objects we will use for plots for both field components for each mode
line_Ex, = ax1.plot([], [],'r', label='$E_x$', lw=2)
line_Hy, = ax1.plot([], [],'b', label='$H_y$')

line_Ey, = ax2.plot([], [],'r', label='$E_y$',lw=2)
line_Hx, = ax2.plot([], [],'b', label='$H_x$')

z_range = np.linspace(0,N-1,N)
dN = int(np.ceil(N/100))

# Setting the axis limits
ax1.set_xlim(0,N)
ax1.set_ylim(-1.5,1.5)

ax2.set_xlim(0,N)
ax2.set_ylim(-1.5,1.5)

# labels
ax1.set_title('$E_x/H_y$-mode')
ax1.legend()
fig.text(0.06, 0.5, 'normalised field values', ha='center', va='center', rotation='vertical')

ax2.set_title('$E_y/H_x$-mode')
ax2.set_xlabel('$z$ (in m)')
ax2.legend()

# Called for each frame to plot the data until the index `frame`
def update(t):
    # step = int(np.ceil(N/(t_final*fps))) # you migh get an index error
    
    fig.suptitle('Number of cells: '+str(N)+ ', Time step '+str(t)+ ' of '+str(T))
    
    # plot E_x, H_y in the top plot 
    line_Ex.set_data(z_range, E_x[:,t]) 
    line_Hy.set_data(z_range, 377*H_y[:,t])
    
    # plot E_y, H_x in the bottom plot 
    line_Ey.set_data(z_range, E_y[:,t]) 
    line_Hx.set_data(z_range, 377*H_x[:,t])
    
    return [line_Ex, line_Hy, line_Ey, line_Hx]

# create the animation object using the figure `fig` by calling the function update for each value in the list `frame`
# Use fps as frames per second, so the delay between each frame is 1000 ms / fps
ani = FuncAnimation(fig, update, frames=range(T), interval = 1000//fps,blit=True)

plt.show()