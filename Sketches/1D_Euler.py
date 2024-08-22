import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation
from matplotlib.patches import Ellipse

def boundary(A):
  
  global Ng
  
  # periodic
  # A[:,  Ng-1] = A[:, -Ng-1]
  # A[:,  Ng-2] = A[:, -Ng-2]
  # A[:, -Ng+0] = A[:,  Ng+0]
  # A[:, -Ng+1] = A[:,  Ng+1]
  
  # A[:,  Ng-1] = 4.
  # A[:,  Ng-2] = 4.
  # A[:, -Ng+1] = 4.
  # A[:, -Ng+2] = 4.
  
  # outflow
  A[:,  Ng-1] = A[:,  Ng-0]
  A[:,  Ng-2] = A[:,  Ng-1]
  
  A[:, -Ng+0] = A[:, -Ng-1]
  A[:, -Ng+1] = A[:, -Ng-0]
  
def primititives2conserved(n, u, p):

  global gamma

  out = np.zeros( (3, n.shape[0]) )
  
  out[0,:] = n
  out[1,:] = n * u
  out[2,:] = p / (gamma-1.) + 0.5 * n * u**2
  
  return out

def conserved2primitives(A):
  
  n =   A[0,:]
  u =   A[1,:] / n
  p = ( A[2,:] - 0.5 * n * u**2 ) * (gamma-1.)
  
  return n, u, p

def init():
  
  global Q
  global x, L
  
  n = np.zeros_like(x)
  u = np.zeros_like(x)
  p = np.zeros_like(x)
  
  n += 1.
  u += 0.
  p += 1.
  
  n[x > 0.5 * L] = 0.125
  u[x > 0.5 * L] = 0.
  p[x > 0.5 * L] = 0.1
  
  Q = primititives2conserved(n,u,p)
  
  boundary(Q)
  
def flux(A):
  
  F = np.zeros_like( A )
  
  n, u, p = conserved2primitives( A )
  E = A[2,:]
  
  F[0,:] = u * n
  F[1,:] = u * n*u + p
  F[2,:] = u * (E + p)
  
  return F
  
def step():
  
  global Q
  global t, dt, dx
  
  n, u, p = conserved2primitives( Q )
  
  # LxF
  # a_L = dx/dt
  # a_R = dx/dt
  
  # Rusanov
  c = np.sqrt( gamma * p / n )
  a = np.abs(u) + c
  a_L = np.maximum( a[Ng:-Ng] , a[Ng-1:-Ng-1] )
  a_R = np.maximum( a[Ng+1:-Ng+1] , a[Ng:-Ng] )
  
  F = flux(Q)
  
  Q[:,Ng:-Ng] += - 0.5 * dt/dx * ( F[:,Ng+1:-Ng+1] - F[:,Ng-1:-Ng-1] ) \
                 + 0.5 * dt/dx * ( a_R * ( Q[:,Ng+1:-Ng+1] - Q[:,Ng:-Ng] ) - a_L * ( Q[:,Ng:-Ng] - Q[:,Ng-1:-Ng-1] ) ) 

  boundary(Q)

  t += dt

''' MAIN '''

L = 1.
N = 2000
cfl = 0.1
Ng = 2

dx = L/(N-1)
t = 0.
t_out = 0.
dt = cfl*dx

gamma = 1.4

x = np.linspace(-Ng*dx,L+Ng*dx,N+2*Ng, endpoint=True)
X,Y = np.meshgrid(x,x)

Q = np.zeros( (3, N+2*Ng) )

init()

while(t < 0.2):
  step()

''' GRAPHICS '''

fig, ax = plt.subplots(1,1, figsize=(10, 10))
line,  = ax.plot( x, Q[0,:], '-')
# line = ax.plot( x[Ng:-Ng], Q[0,Ng:-Ng])
title = "time = {:.2f} s".format(t)
fig.suptitle(title, fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Y')
for val in [0.263, 0.500, 0.685, 0.850]:
  ax.axvline(val, c='k', ls='--', zorder=0)
ax.set_xlim(0., 1.0)
ax.set_ylim(0., 1.1)

# def update(i):
  
#   global Q
#   global t, t_out
  
#   dt_out = 0.05
  
#   # while( t_out < dt_out ):
#     # step()
#     # t_out += dt
#   # t_out -= dt_out
  
#   step()
  
#   line.set_ydata( (Q[0,:]) ) 
  
#   title = "time = {:.2f} s".format(t)
#   fig.suptitle(title, fontsize=16)
  
#   return line, 

# anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

# plt.savefig('Sod_Rusanov')

plt.show()