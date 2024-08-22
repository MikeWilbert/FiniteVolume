import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation

def init():
  
  global x, Q

  p = -np.cos( 2*5.*x * 2.*np.pi/L)

  p[x < 0.] = -1.
  p[x < -0.5 * 0.125*L*0.5] =  1.
  p[x < -0.5 * 0.875*L*0.5] = -1.

  p[x >  0.125*L] = 0.
  p[x < -0.125*L] = 0.

  u = np.zeros_like(p)
  
  Q[0,:] = p
  Q[1,:] = u
  
def step():
  
  global Q
  global A_p, A_m
  global a, dx, dt
  
  F_r = np.matmul( A_p, Q ) + np.matmul( A_m, np.roll(Q, -1) )
  F_l = np.roll(F_r, 1)
  
  Q = Q - dt/dx * ( F_r - F_l )

L = 1.
N = 200
cfl = 1.
c = 1.
Z = 1.

dx = L/(N-1)
dt = cfl * dx/c
a = dt / dx
t = 0.

x = np.linspace(-0.5*L,0.5*L,N, endpoint=False)
Q = np.zeros( (2, N) ) # Q[0]: p, Q[1]: u

A_p = 0.5*c * np.array( [ [  1, Z ], [ 1/Z,  1 ]] )
A_m = 0.5*c * np.array( [ [ -1, Z ], [ 1/Z, -1 ]] )

fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10, 20))
ax1.grid()
ax1.set_xlim(-0.5*L, 0.5*L)
ax1.set_ylim(-1.5, 1.5)
ax2.grid()
ax2.set_xlim(-0.5*L, 0.5*L)
ax2.set_ylim(-1.5, 1.5)

init()

line_1, = ax1.plot(x, Q[0,:])
line_2, = ax2.plot(x, Q[1,:])

def update(i):
  global x, Q, t
  
  step()

  t += dt
  
  title = "time = {:.2f}".format(t)
  fig.suptitle(title)
  
  line_1.set_ydata( Q[0,:] ) 
  line_2.set_ydata( Q[1,:] ) 
  
  return line_1, line_2 

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()

plt.show()