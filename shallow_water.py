import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation

def init():
  
  global x, Q, b 
  global b_start, h_0, g

  A = 2. * h_0

  eta = np.zeros_like(x)
  b   = np.zeros_like(x)
  h   = np.zeros_like(x)
  U   = np.zeros_like(x)
  
  b   += h_0
  # b   += - 0.4 * np.exp( - (x - 0.)**2 / 20. )
  b[x < 20.] += 1./20. * (x[x<20.] - 20.) 
  # b[x < 0.] += - 0.04 * ( 10 - np.abs(x[x<0.] - (-10.) ) )
  # b += - 0.8 * np.exp( - (x - 10.)**2 / 10. )
  # b += - 0.8 * np.exp( - (x - 0.)**2 / 10. )
  # b += - 0.8 * np.exp( - (x + 10.)**2 / 10. )
  
  eta += 2. * np.exp( - (x - 80.)**2 / 20. )
  # eta += A/h_0 * 1./np.cosh( np.sqrt(0.75 * A/h_0**3) * (x - 80.))
  
  H = b
  U += 0.

  Q[0,:] = H
  Q[1,:] = U*H
  
  Q[0,:] += eta
  Q[1,:] += -eta * np.sqrt(g*eta)

  boundary(Q)
  dry(Q)

def boundary(u):

  # outflow 
  u[0,0] = b[0]
  u[0,1] = b[1]
  u[0,-1] = b[-1]
  u[0,-2] = b[-2]
  u[1,0] = 0.
  u[1,1] = 0.
  u[1,-1] = 0.
  u[1,-2] = 0.

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
  
  return ( cfl * dx / c )

def flux(u):
  
  global g
  
  f = np.zeros_like(u)
  
  f[0,:] = u[1,:]
  f[1,:] = u[1,:]**2 / ( u[0,:] + 1.e-12 ) + 0.5 * g * u[0,:]**2
  
  return f

def dry(u):
  
  u[:,:] = np.where( u[0,:] < 1.e-6, 0., u[:,:] )

def step():
  
  global Q, dt, t, g, dx, b
  
  dt = calc_dt()
  
  Q_l = Q[:,:-1]
  Q_r = Q[:,1:]
  
  a = np.where( max_u(Q_l) > max_u(Q_r), max_u(Q_l), max_u(Q_r) )
  
  # LxF
  # F = 0.5 * ( flux( Q_r ) + flux( Q_l ) - dx/dt * ( Q_r - Q_l ) )
  # Rusanov  
  F = 0.5 * ( flux( Q_r ) + flux( Q_l ) - a * ( Q_r - Q_l ) )
  
  F_L = F[:,1:-2]
  F_R = F[:,2:-1]
  
  # Strang splitting
  S = + ( b[3:-1] - b[1:-3]) / (2.*dx) * g * Q[0,2:-2]
  Q[1,2:-2] += 0.5 * dt * S 
  dry(Q)
  Q[:,Ng:-Ng] += - dt * ( F_R - F_L ) / dx
  dry(Q)
  S = + ( b[3:-1] - b[1:-3]) / (2.*dx) * g * Q[0,2:-2]
  Q[1,2:-2] += 0.5 * dt * S 
  dry(Q)
  
  boundary(Q)
  dry(Q)
  
  t += dt

L_l = 20.
L_r = 80
b_start = 50.
h_0 = 1.
g = 9.81

L = L_r + L_l
N = 200
cfl = 0.5

Ng = 2

dx = L/(N-1)
t = 0.
dt = 0.1*dx/(3.)

x = np.linspace(-L_l-Ng*dx,L_r+Ng*dx,N+2*Ng, endpoint=True)
Q = np.zeros( (2,N+2*Ng) )
b = np.zeros( N+2*Ng )

init()

fig, ax1 = plt.subplots(1,1, figsize=(20, 4))
ax1.set_xlim(-L_l, L_r)
ax1.set_ylim(-1.5*h_0, h_0)
ax1.axhline(0.,-L_l, L_r, c='k', alpha=0.1, ls='--')

# line_1, = ax1.plot(x, (Q[0,:]-h_0), c = 'b', alpha=0.5  )
fill_1 = ax1.fill_between(x, (Q[0,:]-b), -b, facecolor = 'xkcd:bright blue')
fill_2 = ax1.fill_between(x, -b, - 1.5 * h_0, facecolor = 'xkcd:light mustard')

def update(i):
  global Q, t
  # global line_1
  global fill_1, fill_2, ax1
  
  step()
  
  title = "time = {:.2f}".format(t)
  fig.suptitle(title)
  
  # line_1.set_ydata( (Q[0,:]-b) )  

  fill_1.remove()
  fill_2.remove()
  fill_1 = ax1.fill_between(x, (Q[0,:]-b), -b, facecolor = 'xkcd:bright blue')
  fill_2 = ax1.fill_between(x, -b, - 1.5 * h_0, facecolor = 'xkcd:light mustard')

  return [fill_1, fill_2]
  # return [line_1 , fill_1, fill_2]

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()