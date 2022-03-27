# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:02:29 2020

@author: fbar
"""


# Importing packages
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib toolkit for 3D plots 
from mpl_toolkits.mplot3d import Axes3D

# Method for animating plots
from matplotlib.animation import FuncAnimation


## Input

# Set the number N of time steps 
N = 100000
# Set a final time 
t_final = 40

# Create time object. We want N number of time steps not counting t = 0
time, dt = np.linspace(0, t_final, N+1, retstep = True)

# initial position and velocity of the point particle
pos_init = np.array([.0, .0, .0])
vel_init = np.array([0.0, 0.0, 0.0 ])

# Mass of point particle (in kg)
m = 1

# Charge of the point particle (in C)
q = 1

# Speed of light in vacuum 
c0 = 299792458

# Electric field strength (in N/C)
def E(pos: np.array, t: float)-> np.array:
    return 200*np.array([1.0, .0, .0])*np.sin(2*np.pi*(t/(5000*dt)+pos[1]/(c0*5000*dt)))

# Magnetic flux density (in N/(Am))
def B(pos: np.array, t: float)-> np.array:
    return 50*np.array([.0, .0, 0.5])*np.sin(2*np.pi*(t/(5000*dt)+pos[1]/(c0*5000*dt)))



# Implementing the Euler method

# Calculates the acceleration
def accel(pos: np.array, vel: np.array, t:float)-> np.array:
    return (q/m)*(E(pos,t) + np.cross(vel, B(pos,t)))

# Set the intial values for position and velocity
pos = pos_init
vel = vel_init

# array for saving the position for each time step for plotting and evaluation
# We think of it as a list with N+1 3D vectors
trajectory = np.empty((N+1, 3))
trajectory[0] = pos

# Calculate the new position and velocity for each time step and save them in the respective arrays
# We have to start with time step 1
for i, t in enumerate(time[1:], 1):
    # Calculation of the trajectory with friction
    pos = pos + vel*dt
    vel = vel + accel(pos, vel, t)*dt
    trajectory[i] = pos

# Output: plot the trajectory of the point charge

fig3D = plt.figure(figsize = (10,8)) 
ax = Axes3D(fig3D) #fig3D.add_subplot(111, projection='3d')

ax.plot(trajectory[:N+1:(N+1)//1000, 0], trajectory[:N+1:(N+1)//1000, 1], trajectory[:N+1:(N+1)//1000, 2], label = 'trajectory')

ax.legend()

ax.set_xlabel("$x$-coordinate")
ax.set_ylabel("$y$-coordinate")
ax.set_zlabel("$z$-coordinate")

# plt.show()

## Animating the curve

# Setting up the plotting figure
anim_fig = plt.figure(figsize=(10,8)) 
anim_ax = Axes3D(anim_fig)

# frames per second
fps = 25

anim_fig.suptitle('Charged particle in an electric and magnetic field. (Euler method)' + '\n'
                  + 'mass: ' + str(m)+ ' kg               charge: ' + str(q) + ' C \n'
                  + 'Initial position: '+ str(pos_init) + ' m               Initial velocity: '+ str(vel_init)+ ' m/s \n'
                  + 'E-field: ' + str(E(pos_init, 5000*dt)) + ' N/C               B-field: '+str(B(pos_init,5000*dt)) +' N/(Am)')

# Saving the line objects we will use for the plot. 
# Note that in 3D it is currently not possible to pass on empty lists as in 2D
line, = anim_ax.plot(trajectory[0:1,0], trajectory[0:1,1], trajectory[0:1,2])

# Setting range and labels of coordinate axes. 

anim_ax.set_xlim3d(trajectory[:,0].min()-5, trajectory[:,0].max()+5)
anim_ax.set_xlabel('$x-coordinate$')
anim_ax.set_ylim3d(trajectory[:,1].min()-5, trajectory[:,1].max()+5)
anim_ax.set_ylabel('$y-coordinate$')
anim_ax.set_zlim3d(trajectory[:,2].min()-5, trajectory[:,2].max()+5)
anim_ax.set_zlabel('$z-coordinate$')

# Caclulate step size for frames as we do not need to plot every frame
step = int(np.ceil(N/(t_final*fps)))
dangle = 360/(t_final*fps)

# Called for each frame to plot the data until the index `frame`
def update(frame):
    # anim_ax.view_init(30,dangle*frame)
    # There is no `set_data` in 3D. You have to pass on the 2D data and then specify 
    # the z-coordinate for matplot lib so it knows how to calculate the 3D projection 
    line.set_data(trajectory[:frame*step,0], trajectory[:frame*step,1])
    line.set_3d_properties(trajectory[:frame*step,2])
    return line, #anim_ax]

# create the animation object using the figure `anim_fig` by calling the function update for each value in the list `frame`
# Use fps as frames per second, so the delay between each frame is 1000 ms / fps
ani = FuncAnimation(anim_fig, update, frames=range(np.around(t_final*fps)), interval = 1000//fps,
                     blit=True)

# ani.save('Lorentz force demo rotating frame.mp4', fps=25)