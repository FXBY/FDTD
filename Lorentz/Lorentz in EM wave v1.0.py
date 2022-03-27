# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:02:29 2020

@author: fbar
"""


# Importing packages
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# Matplotlib toolkit for 3D plots 
from mpl_toolkits.mplot3d import Axes3D

# Method for animating plots
from matplotlib.animation import FuncAnimation


## Input

# Frequency of EM wave
f = 1000

# Set an initial and final time 
t0 = 0
t_final = 5000*1/f

# Set the number N of time steps 
N =  10000 # max(f*t_final*10, 100000)

# Number of frames to skip per animation cycle
step = 20

# Create time object. We want N number of time steps not counting t = 0
time, dt = np.linspace(t0, t_final, N+1, retstep = True)

# Mass of point particle (in kg)
m = 1# 9.10938356e-31

# Charge of the point particle (in C)
q = 1# 1.60217662e-19

# Stiffness for Hooke's force binding particle
k = 1

# Speed of light in vacuum 
c0 = 299792458

# initial position and velocity of the point particle
pos_init = np.array([.0, .0, .0])
vel_init = 0.0000*(c0/np.sqrt(1))*np.array([-1.0, 0, 0 ])

# Amplitude of electric field 
# (corresponds roughly to 1 kW microwave radiation on 1 m^2)
E0 = np.array([1000.0, .0, .0])
B0 = (1/c0)*np.cross(E0,np.array([0,1,0]))

# Electric field strength (in N/C)
def E(pos: np.array, t: float)-> np.array:
    return E0*np.sin(2*np.pi*f*(t+pos[1]/c0))

# Magnetic flux density (in N/(Am))
def B(pos: np.array, t: float)-> np.array:
    return B0*np.sin(2*np.pi*f*(t+pos[1]/c0))


## Main

# Implementing the Runge-Kutta 4th order method from SciPy. For this the 2nd order ODE from
# Newton's Law has to be rewritten as a system of 1st order ODEs

# Calculates the derivative of the corresponding system of 1st order ODEs
def Lorentz(t:float, y:np.array)-> np.array:
    x = y[:3]
    v = y[3:6]
    return np.concatenate((v, (q/m)*(E(x,t) + np.cross(v, B(x,t)))-k/m*x))

# Initial value
val_init = np.concatenate((pos_init, vel_init))

# Apply the RK45 solver from SciPy and calculate the trajectory
sol = solve_ivp(Lorentz, [t0, t_final], val_init, t_eval = time, method = 'Radau')
trajectory = np.asarray([sol.y[0:3,t] for t in range(time.size)])

## Output

# Output: plot the trajectory of the point charge

fig3D = plt.figure(figsize = (10,8)) 
ax = Axes3D(fig3D) #fig3D.add_subplot(111, projection='3d')

# ax.plot(trajectory[:N+1:(N+1)//1000, 0], trajectory[:N+1:(N+1)//1000, 1], trajectory[:N+1:(N+1)//1000, 2], label = 'trajectory')
ax.plot(trajectory[:N+1, 0], trajectory[:N+1, 1], trajectory[:N+1, 2], label = 'trajectory')

ax.legend()

ax.set_xlabel("$x$-coordinate")
ax.set_ylabel("$y$-coordinate")
ax.set_zlabel("$z$-coordinate")

# Setting up the plot of the vector field

# The plot functions for the E and B-field
def Field_plot(x:float, y:float, z:float, t:float, V)-> tuple:
    pos = np.array([x,y,z])
    Res = V(pos,t)
    return (Res[0], Res[1], Res[2])

# Allow V to be applied to each component of a numpy array. This way we will avoid iterated loops.
VecField_plot = np.vectorize(Field_plot)

# Create the grid to calculate and display the vector field on 
min_x, max_x = trajectory[:,0].min(), trajectory[:,0].max()
step_x = (max_x-min_x)/10
min_y, max_y = trajectory[:,1].min(), trajectory[:,1].max()
step_y = (max_y-min_y)/10
min_z, max_z = trajectory[:,2].min() - 1e-10, trajectory[:,2].max()+ 1e-10
step_z = (max_z-min_z)/10

X, Y, Z = np.meshgrid(np.arange(min_x,max_x,step_x), np.arange(min_y,max_y,step_y), np.arange(min_z,max_z,step_z))

# plt.show()

## Animating the curve

# Setting up the plotting figure
anim_fig = plt.figure(figsize=(12,10)) 
anim_ax = Axes3D(anim_fig)

# frames per second
fps = 25

# Saving the line objects we will use for the plot. 
# Note that in 3D it is currently not possible to pass on empty lists as in 2D
line, = anim_ax.plot(trajectory[0:1,0], trajectory[0:1,1], trajectory[0:1,2])

# Calculate the Vector field at each point on the grid using the vectorised version of V
# As the plotting functions of matplotlib want separate x, y and z-components we use tuple unpacking
E_X, E_Y, E_Z = VecField_plot(X,Y,Z,0,E) 
B_X, B_Y, B_Z = VecField_plot(X,Y,Z,0,B) 

# We normalise all of the vector to unit vectors to be able to see the vector field
#q_E = anim_ax.quiver(X, Y, Z, E_X, E_Y, E_Z, length = 1, color='red' ,alpha=0.5, normalize= True)
# q_B = anim_ax.quiver(X, Y, Z, B_X, B_Y, B_Z, color='blue' ,alpha=0.3, normalize= True)

# Setting range and labels of coordinate axes. 

anim_ax.set_xlim3d(min_x, max_x)
anim_ax.set_xlabel('$x-coordinate$')
anim_ax.set_ylim3d(min_y, max_y)
anim_ax.set_ylabel('$y-coordinate$')
anim_ax.set_zlim3d(min_z, max_z)
anim_ax.set_zlabel('$z-coordinate$')

# Caclulate step size for frames as we do not need to plot every frame
# step = int(np.ceil(N/(t_final*fps)))
dangle = 360/(t_final*fps)

# Called for each frame to plot the data until the index `frame`
def update(frame):
    #anim_ax.view_init(30,dangle*frame)
    # There is no `set_data` in 3D. You have to pass on the 2D data and then specify 
    # the z-coordinate for matplot lib so it knows how to calculate the 3D projection 
    line.set_data(trajectory[:frame*step,0], trajectory[:frame*step,1])
    line.set_3d_properties(trajectory[:frame*step,2])
    
    # Plotting the E and B-Field (Does not work)
    # new_seg = []
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         for k in range(X.shape[2]):
    #             E_x, E_y, E_z = Field_plot(X[i,j,k], Y[i,j,k], Z[i,j,k], frame*step, E)
    #             new_seg += [[[X[i,j,k], Y[i,j,k], Z[i,j,k]],[E_x,E_y,E_z]]]
    # q_E.set_segments(new_seg)
    
    # B_X, B_Y, B_Z = VecField_plot(X,Y,Z,frame*step,B)
    #q_B.remove()
    # q_B = anim_ax.quiver(X, Y, Z, B_X, B_Y, B_Z, color='blue' ,alpha=0.3, normalize= True)
    return line, #q_E] #anim_ax]

# create the animation object using the figure `anim_fig` by calling the function update for each value in the list `frame`
# Use fps as frames per second, so the delay between each frame is 1000 ms / fps
ani = FuncAnimation(anim_fig, update, frames=range(N//step), interval = 1000//fps,
                     blit=True)

anim_fig.suptitle(f'Harmonically bound charged particle (stiffness = {k:.1f} N/m) in a plane EM-wave. (Runge-Kutta method)' + '\n'
                  + f'mass: {m:.2E} kg               charge: {q:.2E} C \n'
                  + f'Initial position: {pos_init} m               Initial velocity: {vel_init} m/s \n'
                  + f'Plane EM-wave with amplitudes $E=${E0} N/C, $B=${B0} N/(Am)'+'\n'
                  + f' traveling in the $y$-direction with frequency $f=${f:.1f} Hz')

# ani.save('Lorentz force demo - free charge in a radio wave.mp4', fps=fps)