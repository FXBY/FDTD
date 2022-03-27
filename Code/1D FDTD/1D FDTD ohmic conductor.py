# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:06:08 2020

@author: fbar
"""


# importing packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Method for animating plots
from matplotlib.animation import FuncAnimation

# For jupyter notebooks: to display animations inline
# IPython.display import HTML

#### Constants (in SI units)

# Speed of light in vacuum 
c0 = 299792458

# magnetic permeability
mu0 = 1.25663706212e-6

# (derived) electric permittivity
ep0 = 1/(c0**2*mu0)

# vacuum impedance
eta0 = np.sqrt(mu0/ep0)

#### Simulation parameters 

## Yee grid parameters
# Number of cells 
N = 200

# spatial stepsize
# only dz relevant in 1D FDTD
dx = 0 
dy = 0
dz = 1

## Time parameters

# Mumber of time steps
T = 250

# time step
# Courant stability condition: in vacuum field travels max a cell per time step
dt = np.sqrt(dx**2 + dy**2 + dz**2)/(1*c0)

# Calculate update coefficient
m = c0*dt

# Relative electric permittivity and magnetic permeability on the grid
# We are considering EM fields in vacuum only and isotropic homogeneous materials 
# with constant electric and magnetic properties in time as well as conductivity.
# Note that conductivity relates to the normalised D-field and is a flux conductivity
# so literature value has to be divided by ep_R
ep_R = np.ones(N)
mu_R = np.ones(N)
sigma = np.zeros(N)

## modelling a slabs of materials
material_name = '(conductivity $\sigma = 0.1\, A(Vm)^{-1})$' 

# number of slabs
num_slabs = 1

# initial and final cells of slabs
slab = np.zeros((num_slabs, 2), dtype=int)
thickness = 40
s0 = 80

for k in range(num_slabs):
    slab[k,0] = s0 + k*thickness
    slab[k,1] = s0 + (k+1)*thickness

# setting material constants on grid
ep_R[slab[0,0]:(slab[0,1]+1)] = 1
mu_R[slab[0,0]:(slab[0,1])] = 1
# Remember to divide literature sigma by ep_R and to leave the other factors below
# so it is correctly normalised. This is not done in the code!
sigma[slab[0,0]:(slab[0,1]+1)] = 0.1*m*eta0/(2*1)

# ep_R[slab[1,0]:(slab[1,1]+1)] = 2.31
# mu_R[slab[1,0]:(slab[1,1])] = 2*2.31

# ep_R[slab[2,0]:(slab[2,1]+1)] = 4*2.31
# mu_R[slab[2,0]:(slab[2,1])] = 1

# Blurring the boundary cells
#ep_R[slab_0-1] = (ep_R[slab_0-1] + ep_R[slab_0])/2
#ep_R[slab_1+1] = (ep_R[slab_1+1] + ep_R[slab_1])/2
#mu_R[150] = (mu_R[149] + mu_R[150])/2



## Initialising EM fields

# (normalised) Electric field vectors
E = np.zeros((N,T,3))
D = np.zeros((N,T,3))

# (normalised) Magnetic field vectors
H = np.zeros((N,T,3))
B = np.zeros((N,T,3))


## parameters for Gaussian pulse as a source


## Calculate source for TF/SF

# cell where source is injected
k0 = 30

# E and B source field amplitude vectors
E_0 = 1.2*np.array([1,1,0])
H_0 = 1/mu_R[k0]*np.cross(np.array([0,0,1]), E_0)

# Defining the source function
def g(t:float)->float: 
    # Morlet wavelet
    # return np.exp(-0.5*((t - 5*20)/20)**2)*np.cos(2*np.pi*t/20)
    # Ricker wavelet
    # return (1-((t - 6*5)/5)**2)*np.exp(-0.5*((t - 6*5)/5)**2)
    # Gaussian pulse
    return np.exp(-0.5*((t - 6*5)/5)**2)
    # Sine wave eased in with a sigmoid function
    #return 0.5/(1 + np.exp(-100*(t - 43)))*np.sin(2*np.pi*t/43)
    # Ordinary sine wave
    #return np.sin(2*np.pi*t/25)

# TF/SF E-field correction at cell k0
E_src = np.array([ E_0*g(t) for t in range(T)])

# TF/SF H-Field correction at cell k0 - 1
H_src = np.array([ H_0*g(t - (-0.5)*np.sqrt(ep_R[k0]*mu_R[k0]) + 0.5) for t in range(T)])

#### Main FDTD Loop

# initialise curl vectors
curlE = np.zeros(3)
curlH = np.zeros(3)

#### Main FDTD loop

## Main Loop over time
for t in range(T-1):
    
    ## Update B and H fields
    
    # Loop over cells apart from right boundary
    for k in range(N-1):
        # Calculate numerical curls of E-Field
        curlE[0] = -(E[k+1,t,1] - E[k,t,1])/dz
        curlE[1] = (E[k+1,t,0] - E[k,t,0])/dz
        
        # Update B-field from induction law
        B[k,t+1] = B[k,t] - m*curlE
        
        # Update H-field
        H[k,t+1] = 1/mu_R[k]*B[k,t+1]
    
    # TF/SF correction for B and H
    B[k0-1,t+1] += E_src[t]*(m/dz)
    H[k0-1,t+1] = 1/mu_R[k0-1]*B[k0-1,t+1]
    
    # ABC on the right
    B[N-1,t+1] = B[N-2,t]
    H[N-1,t+1] = H[N-2,t]
    
    # Dirichlet boundary condition for E-field
    # Calculate numerical curls of E-Field
    # curlE[0] = -(0 - E[N-1,t,1])/dz
    # curlE[1] = (0 - E[N-1,t,0])/dz
        
    # Update B-field from induction law
    # B[N-1,t+1] = B[N-1,t]- m*curlE
    
    # Update H-field
    # H[N-1,t+1] = 1/mu_R[N-1]*B[N-1,t+1]
    
    
    ## Update D and E field
    
    # Dirichlet boundary condition for H-Field
    # Calculate numerical curl of H-Field
    # curlH[0] = -(H[0,t+1,1] - 0)/dz
    # curlH[1] = (H[0,t+1,0] - 0)/dz

    # Update D-Field from Ampere-Maxwell
    # D[0,t+1] = D[0,t] + m*curlH
       
    # Update E-field
    # E[0,t+1] = 1/ep_R[0]*D[0,t+1]
        
    # Loop over cells apart from left boundary
    for k in range(1,N): 
        # Calculate numerical curls of H-Field
        curlH[0] = -(H[k,t+1,1] - H[k-1,t+1,1])/dz
        curlH[1] = (H[k,t+1,0] - H[k-1,t+1,0])/dz

        # Update D-Field from Ampere-Maxwell
        D[k,t+1] = ((1-sigma[k])/(1+sigma[k]))*D[k,t] + curlH*(m/(1+sigma[k]))
        
        # Update E-field
        E[k,t+1] = 1/ep_R[k]*D[k,t+1]
    
    # TF/SF correction for D and E
    D[k0,t+1] -= 1/ep_R[k0]*H_src[t]*(m/dz)
    E[k0,t+1] = 1/ep_R[k0]*D[k0,t+1]
    
    # ABC on the left
    D[0,t+1] = D[1,t]
    E[0,t+1] = E[1,t]
    
#### Output    
    
# Setting up the plotting figure
fig, (ax1, ax2) = plt.subplots(2,1,figsize = (12,8))

# frames per second
fps = 25

# Saving the line2D objects we will use for plots for both field components for each mode
line_Ex, = ax1.plot([], [],'r', label='$E_x$', lw=2)
line_Hy, = ax1.plot([], [],'b', label='$H_y$')

line_Ey, = ax2.plot([], [],'r', label='$E_y$',lw=2)
line_Hx, = ax2.plot([], [],'b', label='$H_x$')

# Setting scale and range of coordinate axis. 
# Find minimum and maximum x and y values for the trajectory without friction
#(min_x1, max_x1) = int(np.floor(np.min(trajectory_norm[:,0]))), int(np.ceil(np.max(trajectory_norm[:,0])))
#(min_y1, max_y1) = int(np.floor(np.min(trajectory_norm[:,1]))), int(np.ceil(np.max(trajectory_norm[:,1])))

# Find minimum and maximum x and y values for the trajectory with friction
#(min_x2, max_x2) = int(np.floor(np.min(trajectory[:,0]))), int(np.ceil(np.max(trajectory[:,0])))
#(min_y2, max_y2) = int(np.floor(np.min(trajectory[:,1]))), int(np.ceil(np.max(trajectory[:,1])))

#anim_ax.set_xlim(np.minimum(min_x1, min_x2), np.maximum(max_x1, max_x2))
#anim_ax.set_ylim(np.minimum(min_y1, min_y2), np.maximum(max_y1, max_y2))

# Datapoints for z-axis of length d with N number of cells 
z_range = np.linspace(0,N-1,N)
# dN = int(np.ceil(N/100))

# Setting the axis limits
ax1.set_xlim(0,N)
ax1.set_ylim(-1.5,1.5)

ax2.set_xlim(0,N)
ax2.set_ylim(-1.5,1.5)

# Plotting the locations of the source and the slabs
ax1.axvline(k0, ls='--')
ax2.axvline(k0, ls='--')

norm = mpl.colors.Normalize(vmin=-2, vmax=2*num_slabs)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

for k in range(num_slabs):
    ax1.fill_between(z_range[slab[k,0]:(slab[k,1]+1)],-1.5,1.5, color=cmap.to_rgba(k))
    ax1.annotate(r' $\varepsilon_r =$ ' + str(ep_R[slab[k,0]]) + '\n' 
                 + r' $\mu_r =$ '+ str(mu_R[slab[k,0]]), (slab[k,0],-1.3))
    ax2.fill_between(z_range[slab[k,0]:(slab[k,1]+1)],-1.5,1.5, color=cmap.to_rgba(k))
    ax2.annotate(r' $\varepsilon_r =$ ' + str(ep_R[slab[k,0]]) + '\n' 
                 + r' $\mu_r =$ '+ str(mu_R[slab[k,0]]), (slab[k,0],-1.3))
# labels
plt.gcf().canvas.set_window_title('1D FDTD simulation. Number of cells: '+str(N)
                                  + ', Time steps: '+str(T)
                                  + ', Material: '+material_name)
fig.suptitle('1D FDTD simulation with ohmic loss. Number of cells: '+str(N)
                                  + ', Time steps: '+str(T) + '\n'
                                  + 'Material: '+material_name) 

ax1.set_title('$E_x/H_y$-mode')
ax1.legend()
fig.text(0.06, 0.5, r'normalised field values $\tilde{E}$ and $\tilde{H}$ in $\sqrt{N}/m$', ha='center', va='center', rotation='vertical')

ax2.set_title('$E_y/H_x$-mode')
ax2.set_xlabel('cell number in $z$-direction')
ax2.legend()

time_count = ax1.annotate('Time step '+str(T)+ ' of '+str(T), (N//5,1.2), ha='center')

# Called for each frame to plot the data until the index `frame`
def update(t):
    
    time_count.set_text('Time step '+str(t)+ ' of '+str(T))
    
    # plot E_x, H_y in the top plot 
    line_Ex.set_data(z_range, E[:,t,0]) 
    line_Hy.set_data(z_range, H[:,t,1])
    
    # plot E_y, H_x in the bottom plot 
    line_Ey.set_data(z_range, E[:,t,1]) 
    line_Hx.set_data(z_range, H[:,t,0])
    
    return [line_Ex, line_Hy, line_Ey, line_Hx, time_count]

# create the animation object using the figure `fig` by calling the function update for each value in the list `frame`
# Use fps as frames per second, so the delay between each frame is 1000 ms / fps
ani = FuncAnimation(fig, update, frames=range(T), interval = 1000//fps,blit=True)

#plt.show()
#ani.save('Ohmic material with small penetration depth.mp4', fps=25)