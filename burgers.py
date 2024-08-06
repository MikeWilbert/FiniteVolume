import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation

def init():
  
  global x, Q

  # rarefaction
  Q += -1.
  Q[N//2+1:] = 2.
  
  # shock
  # Q += 2.
  # Q[N//2+1:] = -1.
  
  # shock
  # Q = np.sin( x * 2.*np.pi/L + 0.5* 2.*np.pi)
  
def boundary():
  
  global Q
  global bc
  
  # periodic
  if( bc[0] == 'p' or bc[1] == 'p' ):
    Q[0  ] = Q[N]
    Q[N+1] = Q[1]
  
  # outflow
  if( bc[0] == 'o' ):
    Q[0  ] = Q[1]
    
  if( bc[1] == 'o' ): 
    Q[N+1] = Q[N]
  
  # reflecting
  if( bc[0] == 'r' ):
    Q[1,0  ] = -Q[1,1]
    
  if( bc[1] == 'r' ):
    Q[1,N+1] = -Q[1,N]

def flux( u ):
  
  return ( 0.5 * u**2 )

def step():
  
  global Q
  global A_p, A_m
  global dx, dt, t, c
  
  dt = calc_dt()
  
  boundary()
  
  
  
  # LxF
  # a = dx/dt
  # F = 0.5 * ( flux( Q_p ) + flux( Q_j ) - a * ( Q_p - Q_j ) )
  
  # Rusanov
  # a = np.fmax( np.abs(Q_p), np.abs(Q_j) )
  # F = 0.5 * ( flux( Q_p ) + flux( Q_j ) - a * ( Q_p - Q_j ) )
  
  # Upwind
  # s = 0.5 * ( Q_j + Q_p )
  # Q_s = np.where( s > 0. , Q_j, Q_p )
  
  # Q_r = np.where( Q_p < 0., Q_p, 0. )
  # Q_r = np.where( Q_j > 0., Q_j, Q_r )
  
  # Q_riemann = np.where( Q_p > Q_j, Q_r, Q_s )
  
  # F = flux( Q_riemann )
  
  # F_l = F[0:N]
  # F_r = F[1:N+1]
  # Q[1:N+1] = Q[1:N+1] - dt/dx * ( F_r - F_l ) 
  
  # KT
  
  Q_m = Q[0:N  ] # N
  Q_j = Q[1:N+1] # N
  Q_p = Q[2:N+2] # N
  
  a = np.fmax( np.abs(Q[1:N+2]), np.abs(Q[0:N+1]) ) # N+1
  
  sigma = ( ( Q_p - Q_m ) / ( 2.*dx ) ) # N
  
  u_p = Q_p - 0.5*dx * sigma
  u_m = Q_j + 0.5*dx * sigma

  # H = 0.5 * ( flux( u_p ) + flux( u_m ) ) - 0.5 * a * ( u_p - u_m )
  # H_m = H[0:N]
  # H_p = H[1:N+1]

  # Q[1:N+1] += - dt/dx * ( H_p - H_m )


  t += dt

def calc_dt():
  
  tau = cfl*dx / np.amax(Q)

  return tau

L = 6.
N = 200
cfl = 0.5
bc = ('p', 'p')

dx = L/(N-1)
t = 0.
dt = 0.

x = np.linspace(-0.5*L-dx,0.5*L+dx,N+2, endpoint=True)
Q = np.zeros( N+2 )

fig, ax1 = plt.subplots(1,1, figsize=(10, 20))
ax1.set_xlim(-0.5*L, 0.5*L)
ax1.set_ylim(-1.5, 2.5)

init()

line_1, = ax1.plot(x[1:N+1], Q[1:N+1], c = 'xkcd:ocean blue', marker='*')
fill_1 = ax1.fill_between(x[1:N+1], Q[1:N+1], -1.5)

def update(i):
  global Q, t
  global line_1, line_2, fill_1, ax1
  
  step()
  
  title = "time = {:.2f}".format(t)
  fig.suptitle(title)
  
  line_1.set_ydata( Q[1:N+1] )  

  fill_1.remove()
  fill_1 = ax1.fill_between(x[1:N+1], Q[1:N+1], -1.5, facecolor = 'xkcd:bright blue')
  
  return [line_1 , fill_1]

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()