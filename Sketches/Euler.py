import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation


def boundary(A):
  
  global Ng
  
  #periodic
  A[:,  Ng-1, Ng:-Ng] = A[:, -Ng-1, Ng:-Ng]
  A[:,  Ng-2, Ng:-Ng] = A[:, -Ng-2, Ng:-Ng]
  A[:, -Ng+0, Ng:-Ng] = A[:,  Ng+0, Ng:-Ng]
  A[:, -Ng+1, Ng:-Ng] = A[:,  Ng+1, Ng:-Ng]
  
  A[:, Ng:-Ng,  Ng-1] = A[:, Ng:-Ng, -Ng-1]
  A[:, Ng:-Ng,  Ng-2] = A[:, Ng:-Ng, -Ng-2]
  A[:, Ng:-Ng, -Ng+0] = A[:, Ng:-Ng,  Ng+0]
  A[:, Ng:-Ng ,-Ng+1] = A[:, Ng:-Ng,  Ng+1]
  
  A[:, Ng-1,  Ng-1] = A[:, -Ng-1,  -Ng-1]
  A[:, Ng-2,  Ng-1] = A[:, -Ng-2,  -Ng-1]
  A[:, Ng-1,  Ng-2] = A[:, -Ng-1,  -Ng-2]
  A[:, Ng-2,  Ng-2] = A[:, -Ng-2,  -Ng-2]
  
  A[:, Ng-1, -Ng+0] = A[:, -Ng-1, Ng+0]
  A[:, Ng-2, -Ng+0] = A[:, -Ng-2, Ng+0]
  A[:, Ng-1, -Ng+1] = A[:, -Ng-1, Ng+1]
  A[:, Ng-2, -Ng+1] = A[:, -Ng-2, Ng+1]
  
  A[:, -Ng+0, Ng-1] = A[:, Ng+0, -Ng-1]
  A[:, -Ng+1, Ng-1] = A[:, Ng+1, -Ng-1]
  A[:, -Ng+0, Ng-2] = A[:, Ng+0, -Ng-2]
  A[:, -Ng+1, Ng-2] = A[:, Ng+1, -Ng-2]
  
  A[:, -Ng+0, -Ng+0] = A[:, Ng+0, Ng+0]
  A[:, -Ng+1, -Ng+0] = A[:, Ng+0, Ng+0]
  A[:, -Ng+0, -Ng+1] = A[:, Ng+0, Ng+0]
  A[:, -Ng+1, -Ng+1] = A[:, Ng+0, Ng+0]
  
def get_conserved( n, ux, uy, p ):
  
  global gamma
  
  A = np.ones( (4,n.shape[0], n.shape[1]) )
  
  A[0,:,:] = n
  A[1,:,:] = n * ux
  A[2,:,:] = n * uy
  A[3,:,:] = ( p / (gamma-1.) +  0.5 * n * (ux**2 + uy**2) )
  
  return A
  
def get_primitives(A):

  n  = A[0,:,:]
  n_inv = 1./n
  ux = A[1,:,:] * n_inv
  uy = A[2,:,:] * n_inv
  p  = ( A[3,:,:] - 0.5 * n * (ux**2 + uy**2) ) * (gamma-1.)
  c  = np.sqrt(gamma*p*n_inv) 

  return n, ux, uy, p, c

def get_dt():
  global Q
  
  pass

def init():
  
  global X, Y
  global Q
  global gamma
  
  n   = np.zeros_like(X)
  ux  = np.zeros_like(X)
  uy  = np.zeros_like(X)
  P   = np.zeros_like(X)
  
  # KH reference
  n = 1. + (np.abs(Y-0.5*L) < 0.25*L)
  ux = -0.5 + (np.abs(Y-0.5*L)<0.25*L)
  # uy = 0.1 * (np.abs(X-0.5*L)<0.25*L)
  # uy += 0.1 * np.sin( 2. * 2.*np.pi * X / ( L + dx ) )
  P += 2.5
  
  Q = get_conserved( n, ux, uy, P )
  boundary(Q)

def flux(A):
  
  global gamma
  
  F_x = np.zeros_like( A )
  F_y = np.zeros_like( A )
  
  n, ux, uy, P, c = get_primitives(A)
  E = A[3,:,:]
  
  # E *= 0.
  # P *= 0.
  
  F_x[0,:,:] = ux * n
  F_x[1,:,:] = ux * n * ux + P
  F_x[2,:,:] = ux * n * uy
  F_x[3,:,:] = ux * ( E + P )
  
  F_y[0,:,:] = uy * n
  F_y[1,:,:] = uy * n * ux
  F_y[2,:,:] = uy * n * uy + P
  F_y[3,:,:] = uy * ( E + P )

  return F_x, F_y

def step():
  
  global t, dt, dx, gamma
  global Q
  
  i_C = np.arange(Ng, N+Ng)
  i_L = i_C - 1
  i_R = i_C + 1
  
  i_fields = np.arange(4)
  
  i_CC = np.ix_( i_fields, i_C, i_C )
  i_CL = np.ix_( i_fields, i_C, i_L )
  i_CR = np.ix_( i_fields, i_C, i_R )
  i_LC = np.ix_( i_fields, i_L, i_C )
  i_RC = np.ix_( i_fields, i_R, i_C )
  
  n, ux, uy, P, c = get_primitives(Q)
  
  F_x, F_y = flux( Q )

  # speeds
  a_x = np.abs(ux) + c
  a_y = np.abs(uy) + c
  
  dt = cfl * dx / np.amax( np.amax(a_x), np )
  
  # a_x *= 0.5
  # a_y *= 0.5
       
  ax_L = np.maximum( a_x[Ng:-Ng, Ng:-Ng], a_x[Ng-1:-Ng-1, Ng:-Ng] )
  ax_R = np.maximum( a_x[Ng:-Ng, Ng:-Ng], a_x[Ng+1:-Ng+1, Ng:-Ng] )
  ay_L = np.maximum( a_y[Ng:-Ng, Ng:-Ng], a_y[Ng:-Ng, Ng-1:-Ng-1] )
  ay_R = np.maximum( a_y[Ng:-Ng, Ng:-Ng], a_y[Ng:-Ng, Ng+1:-Ng+1] )
  
  print(dx/dt)
  print( np.amax(c) )
  print( np.amax(ux) )
  print( np.amax(uy) )
  print( np.amax(ax_R) )
  print( '' )
       
  # Rusanov      
  lamb = dt / dx           
  Q[i_CC] += 0.5 * lamb * ( - ( F_x[i_RC] - F_x[i_LC] + F_y[i_CR] - F_y[i_CL] )       \
                            + ax_R*( Q[i_RC] - Q[i_CC] ) - ax_L*( Q[i_CC] - Q[i_LC] ) \
                            + ay_R*( Q[i_CR] - Q[i_CC] ) - ay_L*( Q[i_CC] - Q[i_CL] ) )
  boundary(Q)

  t += dt

''' MAIN '''

L = 1.
N = 150
cfl = 0.2
Ng = 2

dx = L/N
t = 0.
t_out = 0.
dt = cfl*dx/1.

# f = 2.              # degrees of freedom
# gamma = 1. + 2. / f # heat capacity ratio
gamma = 5./3.

x = np.linspace(-Ng*dx,L+Ng*dx,N+2*Ng, endpoint=True)
X, Y = np.meshgrid( x, x, indexing = 'ij' )

Q = np.zeros( (4, N+2*Ng, N+2*Ng) )

init()
  
fig, ax = plt.subplots(1,1, figsize=(10, 10), dpi=80)
title = "time = {:.2f} s".format(t)
fig.suptitle(title, fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')	

# line,  = ax.plot(x, Q[0,4,:], )
# ax.set_ylim( [1., 2.] )
# ax.axhline(1., c='k', ls='--')
line = ax.pcolormesh(x, x, Q[0,:,:].T, cmap='viridis', vmin=1., vmax=2.)
fig.colorbar(line)
 
def update(i):
  
  global Q
  global t, t_out
  
  dt_out = 0.05
  
  while( t_out < dt_out ):
    step()
    t_out += dt
  t_out -= dt_out

  # for i in range(1):
  #   step()
  
  # line.set_ydata(Q[0,4,:])
  line.set_array(Q[0,:,:].T.flatten())
  
  title = "time = {:.2f} s".format(t)
  fig.suptitle(title, fontsize=16)

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()