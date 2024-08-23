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
  n = np.maximum( n, 1.e-12)
  u =   A[1,:] / n
  p = ( A[2,:] - 0.5 * n * u**2 ) * (gamma-1.)
  p = np.maximum( p, 1.e-12)
  
  return n, u, p

def init():
  
  global Q
  global x, L
  
  # Sod's problem
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

def get_dt():
  
  global Q
  global dx, cfl, gamma
  
  n, u, p = conserved2primitives(Q)
  c = np.sqrt( gamma * p / n )
  
  delta_t = cfl*dx / np.amax( np.abs(u)+c)
  
  return delta_t

def get_reconstruction(A):

  A_l = A[:,Ng-2:-Ng  ]
  A_c = A[:,Ng-1:-Ng+1]
  A_r = A[:,Ng  : ]

  sigma_l = ( A_c - A_l ) / dx
  sigma_r = ( A_r - A_c ) / dx
  
  # van leer  
  sigma = sigma_r * sigma_l * (sigma_r + sigma_l) / ( sigma_r**2 + sigma_l**2 + 1.e-12)
  
  A_p = A[:,Ng-1:-Ng] + 0.5 * dx * sigma[:,:-1]
  A_m = A[:,Ng:-Ng+1] - 0.5 * dx * sigma[:, 1:]

  return A_p, A_m

def get_RHS(A):

  A_p, A_m = get_reconstruction(A)
  
  F_p = flux( A_p )
  F_m = flux( A_m )
  
  n_p, u_p, p_p = conserved2primitives( A_p )
  n_m, u_m, p_m = conserved2primitives( A_m )
  
  c_p = np.sqrt( np.fmax( gamma * p_p / n_p, 0. ) )
  c_m = np.sqrt( np.fmax( gamma * p_m / n_m, 0. ) )
  
  a_p = np.abs(u_p) + c_p
  a_m = np.abs(u_m) + c_m
  
  # LxF
  # a = np.ones_like(u_m) * dx/dt
  # Rusanov
  a = np.maximum( a_p, a_m)
  
  H = - 0.5 * (F_m + F_p - a * ( A_m - A_p ))
  
  RHS = ( H[:,1:] - H[:,:-1] ) / dx
  
  return RHS

def step():
  
  global Q
  global t, dt, dx
  
  Q1 = np.zeros_like(Q)
  
  dt = get_dt()
  
  # Heun
  RHS0 = get_RHS(Q)
  Q1[:,Ng:-Ng] = Q[:,Ng:-Ng] +  dt * RHS0
  boundary(Q1)
  
  RHS1 = get_RHS(Q1)
  Q[:,Ng:-Ng] = Q[:,Ng:-Ng] + dt * 0.5 * ( RHS0 + RHS1 )
  boundary(Q)

  t += dt

''' MAIN '''

L = 1.
N = 1000
cfl = 0.5
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

# step()
# step()

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
for val in [0.263, 0.495, 0.685, 0.850]:
  ax.axvline(val, c='k', ls='--', zorder=0)
ax.set_xlim(0., 1.0)
ax.set_ylim(0., 1.1)

def update(i):
  
  global Q
  global t, t_out
  
  dt_out = 0.01
  
  # while( t_out < dt_out ):
  #   step()
  #   t_out += dt
  # t_out -= dt_out
  
  step()
  
  line.set_ydata( (Q[0,:]) ) 
  
  title = "time = {:.2f} s".format(t)
  fig.suptitle(title, fontsize=16)
  
  return line, 

# anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

# plt.savefig('Sod_KL')

plt.show()