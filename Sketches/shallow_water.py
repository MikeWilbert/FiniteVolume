import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation
from matplotlib.patches import Ellipse

def init():
  
  global x, Q, b 
  global h_0, g

  eta = np.zeros_like(x)
  b   = np.zeros_like(x)
  h   = np.zeros_like(x)
  U   = np.zeros_like(x)
  
  # b   += h_0
  # b[x < 20.] += 1./20. * (x[x<20.] - 20.) 
  # b += - 0.8 * np.exp( - (x - 10.)**2 / 10. )
  # b += - 0.3 * np.exp( - (x - 0.)**2 / 10. )
  
  # b += - 0.2 * np.exp( - (x - 0.)**2 / 10. )
  # b += - 0.16 * np.exp( - (x - 20.)**2 / 10. )
  # b += - 0.25 * np.exp( - (x - 50.)**2 / 10. )
  
  # beach model
  b   += h_0
  b[x < 20.] += h_0*1./40. * (x[x<20.] - 20.) 
  b += - h_0*0.25 * np.exp( - (x - 10.)**2 / 10. )
  b += - h_0*0.45 * np.exp( - (x -  0.)**2 / 10. )
  b += - h_0*0.4 * np.exp( - (x - (-5))**2 / 10. )
  b += -  h_0*0.8 * np.exp( - (x - (-12))**2 / 16. )
  b += - h_0*1.2 * np.exp( - (x - (-20))**2 / 20. )
  
  A = 0.4*h_0
  eta += A/h_0 * 1./np.cosh( np.sqrt(0.75*A/h_0**3) * (x - 35.) )
  # eta[x < 50.] = -h_0
  
  H = b
  U += u_0

  Q[0,:] = H
  Q[1,:] = U*H
  dry(Q)
  
  Q[0,:] += eta
  Q[1,:] += -(eta) * np.sqrt(g*(Q[0,:]+eta))

  boundary(Q)
  dry(Q)

def boundary(u):

  global b, u_0

  # outflow 
  u[0,0] = b[0]
  u[0,1] = b[1]
  u[0,-1] = b[-1]
  u[0,-2] = b[-2]
  u[1,0] = u_0
  u[1,1] = u_0
  u[1,-1] = u_0
  u[1,-2] = u_0

  # periodic
  # u[:,0] = u[:,-2]
  # u[:,1] = u[:,-3]
  
  # u[:,-1] = u[:,3]
  # u[:,-2] = u[:,2]

def max_u(u):
  
  return np.abs( u[1,:] / ( u[0,:] + 1.e-12 ) ) + np.sqrt( u[0,:] * g )

def calc_dt():
  
  global Q, g, dx, cfl
  
  c = np.amax( max_u(Q) )
  
  return 0.005
  # return ( cfl * dx / c )

def flux(u):
  
  global g
  
  f = np.zeros_like(u)
  
  f[0,:] = u[1,:]
  f[1,:] = u[1,:]**2 / ( u[0,:] + 1.e-12 ) + 0.5 * g * u[0,:]**2
  
  return f

def dry(u):
  
  u[0,:] = np.where( u[0,:] < 1.e-5, 1.e-5, u[0,:] )
  
def num_flux( u_m, u_p ):
  
  a = np.where( max_u(u_m) > max_u(u_p), max_u(u_m), max_u(u_p) )
  
  return 0.5 * ( flux( u_p ) + flux( u_m ) - a * ( u_p - u_m ) )

def calc_RHS(u):
  
  global dt, t, g, dx, b, N, n
  
  S = np.zeros( (2,N) )

  S[1,:] = ( b[3:-1] - b[1:-3]) / (2.*dx) * g * Q[0,2:-2] - Q[1,2:-2]*np.abs(Q[1,2:-2]) * Q[0,2:-2]**(-7./6.)*g*n**2 
  
  du_l = (u[:,1:-1] - u[:,0:-2]) / dx
  du_r = (u[:,2:  ] - u[:,1:-1]) / dx
  
  theta = du_l / ( du_r + 1.e-12 )
  phi   = (theta + np.abs(theta)) / (1 + np.abs(theta))
  sigma = phi * du_r
  
  u_p = u[:, 2:-1] - 0.5 * dx * sigma[:,1:  ]
  u_m = u[:, 1:-2] + 0.5 * dx * sigma[:, :-1]
  
  F = num_flux( u_m, u_p )
  
  F_L = F[:, :-1]
  F_R = F[:,1:  ]
  
  return ( - ( F_R - F_L ) / dx + S )

def step():
  
  global Q, dt, t, g, dx, b
  
  Q_1 = np.zeros_like(Q)
  
  dt = calc_dt()

  # Heun
  RHS_0 = calc_RHS(Q)
  Q_1[:,2:-2] = Q[:,2:-2] + dt * RHS_0
  boundary(Q_1)
  dry(Q_1)
  
  RHS_1 = calc_RHS(Q_1)
  Q[:,2:-2] = Q[:,2:-2] + dt * 0.5 * ( RHS_0 + RHS_1 )
  boundary(Q)
  dry(Q)
  
  t += dt

L_l = 20.
L_r = 50.
h_0 = 2.
g = 9.81
n = 0.025 # Manning coefficient
u_0 = 0.

L = L_r + L_l
N = 1600
cfl = 0.5

Ng = 2

dx = L/(N-1)
t = 0.
dt = 0.1*dx/(3.)

x = np.linspace(-L_l-Ng*dx,L_r+Ng*dx,N+2*Ng, endpoint=True)
Q = np.zeros( (2,N+2*Ng) )
b = np.zeros( N+2*Ng )

init()

# thumb nail
# while(t < 2.0):
#   step()

# import matplotlib.ticker as ticker
# fig, ax1 = plt.subplots(1,1, figsize=(10, 10))
# ax1.set_xlim(-10, 25)
# ax1.set_ylim(-h_0, h_0)
# ax1.xaxis.set_major_locator(ticker.NullLocator())
# ax1.yaxis.set_major_locator(ticker.NullLocator())
# from matplotlib.patches import Ellipse
# circle = Ellipse((0, 3), 4.,1., color='xkcd:bright yellow')

fig, ax1 = plt.subplots(1,1, figsize=(11, 6))
# fig, ax1 = plt.subplots(1,1, figsize=(12, 6))
# fig, ax1 = plt.subplots(1,1, figsize=(20, 4))
ax1.set_xlim(-L_l, L_r)
ax1.set_ylim(-1.5*h_0, h_0)
ax1.axhline(0.,-L_l, L_r, c='k', alpha=0.1, ls='--')
circle = Ellipse((0, 3), 2.,1., color='xkcd:bright yellow')
ax1.set_xlabel('distance [m]')
ax1.set_ylabel('height [m]')
title = "time = 0.00 s"
fig.suptitle(title, fontsize=16)

# line_1, = ax1.plot(x, (Q[0,:]-h_0), c = 'b', alpha=0.5  )
fill_1 = ax1.fill_between(x, (Q[0,:]-b), -b, facecolor = 'xkcd:bright blue')
fill_2 = ax1.fill_between(x, -b, - 1.5 * h_0, facecolor = 'xkcd:light mustard')
ax1.set_facecolor('xkcd:light blue')
ax1.add_patch(circle)

def update(i):
  global Q, t
  # global line_1
  global fill_1, fill_2, ax1
  
  for i in range(4):
    step()
  
  print(t)
  
  title = "time = {:.2f} s".format(t)
  fig.suptitle(title, fontsize=16)
  
  # line_1.set_ydata( (Q[0,:]-b) )  

  fill_1.remove()
  fill_2.remove()
  fill_1 = ax1.fill_between(x, (Q[0,:]-b), -b, facecolor = 'xkcd:bright blue')
  fill_2 = ax1.fill_between(x, -b, - 1.5 * h_0, facecolor = 'xkcd:light mustard')

  return [fill_1, fill_2]
  # return [line_1 , fill_1, fill_2]

# anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

# plt.savefig('shallow_water.png', dpi=200, bbox_inches='tight')
# plt.show()

# saving to m4 using ffmpeg writer
anim = animation.FuncAnimation( fig=fig, func=update, frames = 200, blit=True )
# anim = animation.FuncAnimation( fig=fig, func=update, frames = 1500, blit=True )
writervideo = animation.FFMpegWriter(fps=40) 
plt.rcParams['savefig.bbox'] = 'tight' 
anim.save('shallow_water_KT.mp4', writer=writervideo, dpi=300) 
plt.close() 