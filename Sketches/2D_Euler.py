import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation

def boundary(A):
  
  global Ng
  
  # periodic
  # A[:,  Ng-1, :] = A[:, -Ng-1, :]
  # A[:,  Ng-2, :] = A[:, -Ng-2, :]
  # A[:, -Ng+0, :] = A[:,  Ng+0, :]
  # A[:, -Ng+1, :] = A[:,  Ng+1, :]
  # A[:, :,  Ng-1] = A[:, :, -Ng-1]
  # A[:, :,  Ng-2] = A[:, :, -Ng-2]
  # A[:, :, -Ng+0] = A[:, :,  Ng+0]
  # A[:, :, -Ng+1] = A[:, :,  Ng+1]
  
  # outflow
  A[:,  Ng-1, :] = A[:,  Ng-0, :]
  A[:,  Ng-2, :] = A[:,  Ng-1, :]
  A[:, -Ng+0, :] = A[:, -Ng-1, :]
  A[:, -Ng+1, :] = A[:, -Ng-0, :]
  A[:, :,  Ng-1] = A[:, :,  Ng-0]
  A[:, :,  Ng-2] = A[:, :,  Ng-1]
  A[:, :, -Ng+0] = A[:, :, -Ng-1]
  A[:,  :,-Ng+1] = A[:, :, -Ng-0]
  
def primititives2conserved(n, ux, uy, p):

  global gamma

  out = np.zeros( (4, n.shape[0], n.shape[1]) )
  
  out[0,:,:] = n
  out[1,:,:] = n * ux
  out[2,:,:] = n * uy
  out[3,:,:] = p / (gamma-1.) + 0.5 * n * (ux**2 + uy**2)
  
  return out

def conserved2primitives(A):
  
  n  =   A[0,:,:]
  ux =   A[1,:,:] / n
  uy =   A[2,:,:] / n
  p  = ( A[3,:,:] - 0.5 * n * (ux**2+uy**2) ) * (gamma-1.)
  
  return n, ux, uy, p

def init():
  
  global Q
  global X, L
  
  n  = np.zeros_like(X)
  ux = np.zeros_like(X)
  uy = np.zeros_like(X)
  p  = np.zeros_like(X)
  
  # KH
  # n += 1.
  # ux[ np.abs(Y) < 0.25 * L ] = 1.
  # uy += 0.
  # p  += 2.5
  
  # Sod
  n  += 1.
  ux += 0.
  uy += 0.
  p  += 1.
  
  # n [X > 0.5 * L] = 0.125
  # ux[X > 0.5 * L] = 0.
  # uy[X > 0.5 * L] = 0.
  # p [X > 0.5 * L] = 0.1
  
  n [Y > 0.5 * L] = 0.125
  ux[Y > 0.5 * L] = 0.
  uy[Y > 0.5 * L] = 0.
  p [Y > 0.5 * L] = 0.1
  
  Q = primititives2conserved(n,ux,uy,p)
  
  boundary(Q)
  
def flux(A):
  
  Fx = np.zeros_like( A )
  Fy = np.zeros_like( A )
  
  n, ux, uy, p = conserved2primitives( A )
  E = A[3,:]
  
  Fx[0,:,:] = ux * n
  Fx[1,:,:] = ux * n*ux + p
  Fx[2,:,:] = ux * n*0.
  Fx[3,:,:] = ux * (E + p)
  
  Fy[0,:,:] = uy * n
  Fy[1,:,:] = uy * n*0.
  Fy[2,:,:] = uy * n*uy + p 
  Fy[3,:,:] = uy * (E + p)
  
  return Fx, Fy
  
def step():
  
  global Q
  global  iCC,  iRC,  iLC,  iCL,  iCR
  global iiCC, iiRC, iiLC, iiCL, iiCR
  global t, dt, dx
  
  n, ux, uy, p = conserved2primitives( Q )
  
  Fx, Fy = flux(Q)
  
  # LxF
  # ax_L = dx/dt
  # ax_R = dx/dt
  # ay_L = dx/dt
  # ay_R = dx/dt
  
  # Rusanov
  c = np.sqrt( gamma * p / n )
  ax = np.abs(ux) + c
  ay = np.abs(uy) + c
  ax_L = np.maximum( ax[iCC] , ax[iLC] )
  ax_R = np.maximum( ax[iRC] , ax[iCC] )
  ay_L = np.maximum( ay[iCC] , ay[iCL] )
  ay_R = np.maximum( ay[iCR] , ay[iCC] )
  
  Hx_L = 0.5 * ( Fx[iiCC] + Fx[iiLC] \
                 -  ( ax_L * ( Q[iiCC] - Q[iiLC] ) ) )
  Hx_R = 0.5 * ( Fx[iiRC] + Fx[iiCC] \
                 -  ( ax_R * ( Q[iiRC] - Q[iiCC] ) ) )
  
  Hy_L = 0.5 * ( Fy[iiCC] + Fy[iiCL] \
                 -  ( ay_L * ( Q[iiCC] - Q[iiCL] ) ) )
  Hy_R = 0.5 * ( Fy[iiCR] + Fy[iiCC] \
                 -  ( ay_R * ( Q[iiCR] - Q[iiCC] ) ) )
  
  # RHS = - ( Hx_R - Hx_L ) / dx
  RHS =                          - ( Hy_R - Hy_L ) / dx
  # RHS = - ( Hx_R - Hx_L ) / dx - ( Hy_R - Hy_L ) / dx
  Q[:,Ng:-Ng,Ng:-Ng] += dt * RHS

  boundary(Q)

  t += dt

''' MAIN '''

#parameters
L = 1.
N = 200
cfl = 0.1
Ng = 2

dx = L/(N-1)
t = 0.
t_out = 0.
dt = cfl*dx

gamma = 1.4

# fields
x = np.linspace(-Ng*dx,L+Ng*dx,N+2*Ng, endpoint=True)
Y,X = np.meshgrid(x,x)

Q = np.zeros( (4, N+2*Ng, N+2*Ng) )

# index arrays
i_C = np.arange(Ng, N+Ng)
i_L = i_C - 1
i_R = i_C + 1
i_fields = np.arange(4)

iCC = np.ix_( i_C, i_C )
iCL = np.ix_( i_C, i_L )
iCR = np.ix_( i_C, i_R )
iLC = np.ix_( i_L, i_C )
iRC = np.ix_( i_R, i_C )

iiCC = np.ix_( i_fields, i_C, i_C )
iiCL = np.ix_( i_fields, i_C, i_L )
iiCR = np.ix_( i_fields, i_C, i_R )
iiLC = np.ix_( i_fields, i_L, i_C )
iiRC = np.ix_( i_fields, i_R, i_C )

# main loop
init()

while(t < 0.2):
  step()

''' GRAPHICS '''

fig, ax = plt.subplots(1,1, figsize=(10, 10))
# line,  = ax.plot( x, Q[0,:,5], '-')
line,  = ax.plot( x, Q[0,5,:], '-')
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