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

  out = np.zeros( ( (4,) + n.shape) )
  
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
  
  # Kelvin-Helmholtz instability (https://doi.org/10.1016/j.compfluid.2015.04.026)
  # BRAUCHE Interpolation! -> KT!!
  # sigma = 0.001
  # n += 1.
  # ux+= 0.5
  # uy+= 0.1*np.sin(2. * 2.*np.pi * X/L)
  # p += 2.5
  
  # n [ np.abs(Y) < 0.25*L ] +=  1
  # ux[ np.abs(Y) < 0.25*L ] += -1
  
  #2D Riemann problem (Case 3 in https://doi.org/10.1137/S1064827502402120)
  # left lower
  n [ ( X < 0. ) & ( Y < 0. ) ] += 0.138
  ux[ ( X < 0. ) & ( Y < 0. ) ] += 1.206
  uy[ ( X < 0. ) & ( Y < 0. ) ] += 1.206
  p [ ( X < 0. ) & ( Y < 0. ) ] += 0.029
  
  # left upper
  n [ ( X < 0. ) & ( Y >= 0. ) ] += 0.5323
  ux[ ( X < 0. ) & ( Y >= 0. ) ] += 1.206
  uy[ ( X < 0. ) & ( Y >= 0. ) ] += 0.0
  p [ ( X < 0. ) & ( Y >= 0. ) ] += 0.3
  
  # right lower
  n [ ( X >= 0. ) & ( Y < 0. ) ] += 0.5323
  ux[ ( X >= 0. ) & ( Y < 0. ) ] += 0.0
  uy[ ( X >= 0. ) & ( Y < 0. ) ] += 1.206
  p [ ( X >= 0. ) & ( Y < 0. ) ] += 0.3
  
  # right upper
  n [ ( X >= 0. ) & ( Y >= 0. ) ] += 1.5
  ux[ ( X >= 0. ) & ( Y >= 0. ) ] += 0.0
  uy[ ( X >= 0. ) & ( Y >= 0. ) ] += 0.0
  p [ ( X >= 0. ) & ( Y >= 0. ) ] += 1.5
  
  Q = primititives2conserved(n,ux,uy,p)
  
  boundary(Q)
  
def flux(n,ux,uy,p,E):
  
  Fx = np.zeros( ( (4,) + n.shape ) )
  Fy = np.zeros( ( (4,) + n.shape ) )
  
  Fx[0,:,:] = ux * n
  Fx[1,:,:] = ux * n*ux + p
  Fx[2,:,:] = ux * n*uy
  Fx[3,:,:] = ux * (E + p)
  
  Fy[0,:,:] = uy * n
  Fy[1,:,:] = uy * n*ux
  Fy[2,:,:] = uy * n*uy + p 
  Fy[3,:,:] = uy * (E + p)
  
  return Fx, Fy

def get_reconstruction(A):
  
  Ax_l = A[:,Ng-2:-Ng  , Ng:-Ng]
  Ax_c = A[:,Ng-1:-Ng+1, Ng:-Ng]
  Ax_r = A[:,Ng  :     , Ng:-Ng]
  Ay_l = A[:,Ng:-Ng, Ng-2:-Ng  ]
  Ay_c = A[:,Ng:-Ng, Ng-1:-Ng+1]
  Ay_r = A[:,Ng:-Ng, Ng  :     ]
  
  sigma_xl = ( Ax_c - Ax_l ) / dx
  sigma_xr = ( Ax_r - Ax_c ) / dx
  sigma_yl = ( Ay_c - Ay_l ) / dx
  sigma_yr = ( Ay_r - Ay_c ) / dx
  
  # van leer  
  sigma_x = sigma_xr * sigma_xl * (sigma_xr + sigma_xl) / ( sigma_xr**2 + sigma_xl**2 + 1.e-12)
  sigma_y = sigma_yr * sigma_yl * (sigma_yr + sigma_yl) / ( sigma_yr**2 + sigma_yl**2 + 1.e-12)
  
  Ax_p = A[:,Ng-1:-Ng, Ng:-Ng] + 0.5 * dx * sigma_x[:,:-1,:]
  Ax_m = A[:,Ng:-Ng+1, Ng:-Ng] - 0.5 * dx * sigma_x[:, 1:,:]
  Ay_p = A[:,Ng:-Ng, Ng-1:-Ng] + 0.5 * dx * sigma_y[:,:,:-1]
  Ay_m = A[:,Ng:-Ng, Ng:-Ng+1] - 0.5 * dx * sigma_y[:,:, 1:]
  
  return Ax_p, Ax_m, Ay_p, Ay_m

def get_RHS(A):
  
  global  iCC,  iRC,  iLC,  iCL,  iCR
  global iiCC, iiRC, iiLC, iiCL, iiCR
  global dt, dx
  
  A_xp, A_xm, A_yp, A_ym = get_reconstruction(A)
     
  n_xp, ux_xp, uy_xp, p_xp = conserved2primitives( A_xp )
  n_xm, ux_xm, uy_xm, p_xm = conserved2primitives( A_xm )
  n_yp, ux_yp, uy_yp, p_yp = conserved2primitives( A_yp )
  n_ym, ux_ym, uy_ym, p_ym = conserved2primitives( A_ym )
  
  c_xp = np.sqrt( np.fmax( gamma * p_xp / n_xp, 0. ) )
  c_xm = np.sqrt( np.fmax( gamma * p_xm / n_xm, 0. ) )
  c_yp = np.sqrt( np.fmax( gamma * p_yp / n_yp, 0. ) )
  c_ym = np.sqrt( np.fmax( gamma * p_ym / n_ym, 0. ) )
  
  a_xp = np.abs(ux_xp) + c_xp
  a_xm = np.abs(ux_xm) + c_xm
  a_yp = np.abs(uy_yp) + c_yp
  a_ym = np.abs(uy_ym) + c_ym
  
  a_x = np.maximum( a_xp, a_xm)
  a_y = np.maximum( a_yp, a_ym)
  
  Fx_xp, Fy_xp = flux(n_xp,ux_xp,uy_xp,p_xp,A_xp[3,:,:])
  Fx_xm, Fy_xm = flux(n_xm,ux_xm,uy_xm,p_xm,A_xm[3,:,:])
  Fx_yp, Fy_yp = flux(n_yp,ux_yp,uy_yp,p_yp,A_yp[3,:,:])
  Fx_ym, Fy_ym = flux(n_ym,ux_ym,uy_ym,p_ym,A_ym[3,:,:])
  
  Hx = - 0.5 * (Fx_xm + Fx_xp - a_x * ( A_xm - A_xp ))
  Hy = - 0.5 * (Fy_ym + Fy_yp - a_y * ( A_ym - A_yp ))
  
  RHS = ( Hx[:,1:,:] - Hx[:,:-1,:] ) / dx + ( Hy[:,:,1:] - Hy[:,:,:-1] ) / dx
  
  return RHS

# old version
# def get_RHS(A):
  
#   global Q
#   global  iCC,  iRC,  iLC,  iCL,  iCR
#   global iiCC, iiRC, iiLC, iiCL, iiCR
#   global dt, dx
  
#   n, ux, uy, p = conserved2primitives( A )
  
#   Fx, Fy = flux(n,ux,uy,p,Q[3,:,:])
  
#   # Rusanov flux
#   c = np.sqrt( gamma * p / n )
#   ax = np.abs(ux) + c
#   ay = np.abs(uy) + c
#   ax_L = np.maximum( ax[iCC] , ax[iLC] )
#   ax_R = np.maximum( ax[iRC] , ax[iCC] )
#   ay_L = np.maximum( ay[iCC] , ay[iCL] )
#   ay_R = np.maximum( ay[iCR] , ay[iCC] )
  
#   Hx_L = 0.5 * ( Fx[iiCC] + Fx[iiLC] \
#                  -  ( ax_L * ( Q[iiCC] - Q[iiLC] ) ) )
#   Hx_R = 0.5 * ( Fx[iiRC] + Fx[iiCC] \
#                  -  ( ax_R * ( Q[iiRC] - Q[iiCC] ) ) )
  
#   Hy_L = 0.5 * ( Fy[iiCC] + Fy[iiCL] \
#                  -  ( ay_L * ( Q[iiCC] - Q[iiCL] ) ) )
#   Hy_R = 0.5 * ( Fy[iiCR] + Fy[iiCC] \
#                  -  ( ay_R * ( Q[iiCR] - Q[iiCC] ) ) )
  
#   RHS = - ( Hx_R - Hx_L ) / dx - ( Hy_R - Hy_L ) / dx
  
#   return RHS

def get_dt():
  
  global Q
  global cfl, dx, gamma
  
  n, ux, uy, p = conserved2primitives( Q )
  c = np.sqrt( gamma * p / n)
  
  delta_t = cfl*dx / np.amax( np.maximum( np.abs(ux) + c,  np.abs(uy) + c ) )
  
  return delta_t
  
def step():
  
  global Q
  global t, dt
  
  dt = get_dt()
  
  Q1 = np.zeros_like(Q)
  
  # Euler
  RHS0 = get_RHS(Q)
  Q[:,Ng:-Ng,Ng:-Ng] = Q[:,Ng:-Ng,Ng:-Ng] +  dt * RHS0
  boundary(Q)
  
  # Heun
  # RHS0 = get_RHS(Q)
  # Q1[:,Ng:-Ng,Ng:-Ng] = Q[:,Ng:-Ng,Ng:-Ng] +  dt * RHS0
  # boundary(Q1)
  
  # RHS1 = get_RHS(Q1)
  # Q [:,Ng:-Ng,Ng:-Ng] = Q[:,Ng:-Ng,Ng:-Ng] +  dt * 0.5 * ( RHS0 + RHS1 )
  # boundary(Q)

  t += dt

''' MAIN '''

#parameters
L = 1.
N = 200
cfl = 0.5
Ng = 2

dx = L/(N-1)
t = 0.
t_out = 0.
dt = cfl*dx

gamma = 1.4

# fields
x = np.linspace(-Ng*dx,L+Ng*dx,N+2*Ng, endpoint=True)  -0.5 * L
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

# while(t < 0.4):
#   step()
#   print(t)

''' GRAPHICS '''

fig, ax = plt.subplots(1,1, figsize=(10, 10))
n, ux, uy, p = conserved2primitives( Q )
pcm = ax.pcolormesh( x[Ng:-Ng], x[Ng:-Ng], p[Ng:-Ng, Ng:-Ng].T, cmap='jet', vmin=0., vmax=1.71 )
cnt = ax.contour(X[Ng:-Ng, Ng:-Ng],Y[Ng:-Ng, Ng:-Ng],n[Ng:-Ng, Ng:-Ng], 32, colors='k', vmin=0.16, vmax=1.71, linewidths=1.)
fig.colorbar(pcm)
ax.set_aspect('equal')
title = "time = {:.2f} s".format(t)
fig.suptitle(title, fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Y')

def update(i):
  
  global Q
  global t, t_out
  global pcm, cnt
  
  dt_out = 0.05
  
  while( t_out < dt_out ):
    step()
    t_out += dt
  t_out -= dt_out
  
  # step()
  
  n, ux, uy, p = conserved2primitives( Q )
  
  pcm.set_array( p[Ng:-Ng, Ng:-Ng].T ) 
  for coll in cnt.collections:
    coll.remove()
  cnt = ax.contour(X[Ng:-Ng, Ng:-Ng],Y[Ng:-Ng, Ng:-Ng],n[Ng:-Ng, Ng:-Ng], 32, colors='k', vmin=0.16, vmax=1.71, linewidths=1.)
  
  title = "time = {:.2f} s".format(t)
  fig.suptitle(title, fontsize=16)

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()