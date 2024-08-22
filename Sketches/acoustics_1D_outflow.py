import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation

def init():
  
  global x, Q

  r_1 = np.array( [ -1, 1./Z ] )

  # testing
  # p = -np.cos( 2*5.*x * 2.*np.pi/L)

  # p[x < 0.] = -1.
  # p[x < -0.5 * 0.125*L*0.5] =  1.
  # p[x < -0.5 * 0.875*L*0.5] = -1.

  # p[x >  0.125*L] = 0.
  # p[x < -0.125*L] = 0.
  
  # Gauss
  p = 0.5 * np.exp( - ( x / (0.1*0.5*L) )**2 )
  
  # zero
  # p = np.zeros(N+2)
  
  # left & right going wave
  # Q[0,:] = 2.*p
  # Q[1,:] = np.zeros_like(p)
  
  # right going wave
  Q[0,:] = p
  Q[1,:] = p / Z
  
def boundary():
  
  global Q
  global bc
  global t
  
  A = 0.5
  lam = 0.25 * 100
  f = c / lam
  phi = 0.

  source =  A * np.sin( f * 2.*np.pi*(t+0.5*dt ) + 2.*np.pi*phi )
  
  # periodic
  if( bc[0] == 'p' or bc[1] == 'p' ):
    Q[:,0  ] = Q[:,N]
    Q[:,N+1] = Q[:,1]
  
  # outflow
  if( bc[0] == 'o' ):
    Q[:,0  ] = Q[:,1]
    
  if( bc[1] == 'o' ): 
    Q[:,N+1] = Q[:,N]
  
  # reflecting
  if( bc[0] == 'r' ):
    Q[0,0  ] =  Q[0,1]
    Q[1,0  ] = -Q[1,1]
    
  if( bc[1] == 'r' ):
    Q[0,N+1] =  Q[0,N]
    Q[1,N+1] = -Q[1,N]
    
  # half-reflecting
  refl = 0.5
  
  if( bc[0] == 'h' ):
    Q[0,0  ] =  refl*Q[0,1]
    Q[1,0  ] = -refl*Q[1,1]
    
  if( bc[1] == 'h' ):
    Q[0,N+1] =  refl*Q[0,N]
    Q[1,N+1] = -refl*Q[1,N]
  
  # source
  if( bc[0] == 's' ):

    source =  A * np.sin( f * 2.*np.pi*(t+0.5*dt ) )

    p = Q[0,1]
    u = Q[1,1]
  
    W_2 = 1./(2.*Z) * ( +p + Z*u )
  
    r_2 = np.array( [ +Z ,1 ] )
    
    Q[:,0] = Q[:,1] + ( source / Z - W_2 ) * r_2
    
  if( bc[1] == 's' ):

    source =  A * np.sin( f * 2.*np.pi*( t+0.5*dt ) + 2.*np.pi*phi )

    p = Q[0,N]
    u = Q[1,N]
  
    W_1 = 1./(2.*Z) * ( -p + Z*u )
  
    r_1 = np.array( [ -Z, 1 ] )
    
    Q[:,N+1] = Q[:,N] + ( source / Z - W_1 ) * r_1

  # vibrating wall
  if( bc[0] == 'w' ):

    source =  A * np.sin( f * 2.*np.pi*( t+0.5*dt ) ) / Z
    
    Q[0,0] = Q[0,1]
    Q[1,0] = 2.*source - Q[1,1]
  
  # changing impedancy
  if( bc[0] == 'z' ):
    
    Q[0,0] = Q[1,0]
    Q[1,0] = 0.
  
def step():
  
  global Q
  global A_p, A_m
  global a, dx, dt, t, c
  
  boundary()
  
  Q_j = Q[:, 0:N+1]
  Q_p = Q[:, 1:N+2]
  F = np.matmul( A_p, Q_j ) + np.matmul( A_m, Q_p )
  
  F_l = F[:,0:N]
  F_r = F[:,1:N+1]
  Q[:, 1:N+1] = Q[:, 1:N+1] - dt/dx * ( F_r - F_l ) 

  t += dt

L = 100.
N = 200
cfl = 0.9
c = 330.
Z = 420.
bc = ('r', 'r')

dx = L/(N-1)
dt = cfl * dx/c
a = dt / dx
t = 0.

x = np.linspace(-0.5*L-dx,0.5*L+dx,N+2, endpoint=True)
Q = np.zeros( (2, N+2) ) # Q[0]: p, Q[1]: u

A_p = 0.5*c * np.array( [ [  1, Z ], [ 1/Z,  1 ]] )
A_m = 0.5*c * np.array( [ [ -1, Z ], [ 1/Z, -1 ]] )

fig, ax1 = plt.subplots(1,1, figsize=(10, 20))
# fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10, 20))
# ax1.grid()
ax1.set_xlim(-0.5*L, 0.5*L)
ax1.set_ylim(-1.5, 1.5)
# ax2.grid()
# ax2.set_xlim(-0.5*L, 0.5*L)
# ax2.set_ylim(-1.5, 1.5)

init()

line_1, = ax1.plot(x[1:N+1], Q[0,1:N+1], c = 'xkcd:ocean blue')
# line_2, = ax2.plot(x[1:N+1], Q[1,1:N+1])

fill_1 = ax1.fill_between(x[1:N+1], Q[0,1:N+1], -1.5)

def update(i):
  global Q, t
  global line_1, line_2, fill_1, ax1
  
  step()
  
  title = "time = {:.2f}".format(t)
  fig.suptitle(title)
  
  line_1.set_ydata( Q[0,1:N+1] ) 
  # line_2.set_ydata( Q[1,1:N+1] ) 

  fill_1.remove()
  fill_1 = ax1.fill_between(x[1:N+1], Q[0,1:N+1], -1.5, facecolor = 'xkcd:bright blue')
  
  return [line_1 , fill_1]
  # return [line_1 , fill_1] , line_2

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()