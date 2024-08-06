import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation

def init():
  
  global x,y

  y = -np.cos(2.*x * 2.*np.pi/L)

  y[x < 0.] = -1.
  y[x < -0.25*L*0.5] =  1.
  y[x < -0.75*L*0.5] =  -1.

def minmod(a, b):
  
  c = np.where( np.abs(a) < np.abs(b), a, b )
  c[a*b < 0.] = 0.

  return c

def minmod3(a, b, c):
  
  tmp = np.where( np.abs(a  ) < np.abs(b), a  , b )
  d   = np.where( np.abs(tmp) < np.abs(c), tmp, c )
  
  d[a*b < 0.] = 0.
  d[a*c < 0.] = 0.
  d[b*c < 0.] = 0.

  return d

def phi(theta):
  global method
  
  if( method == 'upwind' ):
    return 0.
  
  elif( method == 'LxW' ):
    return 1.
  
  elif( method == 'BW' ):
    return theta
  
  elif( method == 'Fromm' ):
    return 0.5 * ( 1 + theta)
  
  elif( method == 'minmod' ):
    return minmod( 1, theta )
  
  elif( method == 'MC' ):
    return minmod3( 0.5*(1+theta) , 2., 2*theta )
  
  elif( method == 'vanLeer' ):
    return ( theta + np.abs(theta) ) / ( 1 + np.abs(theta) )

def step():
  
  global y
  global a, dx, dt, c
  
  theta = ( y - np.roll(y, +1) ) / ( ( np.roll(y, -1) - y ) +  1.e-30 )
  
  F_r = c * y + 0.5 * c * ( 1 - c*dt/dx ) * ( np.roll(y, -1) - y ) * phi( theta )
  F_l = np.roll(F_r, 1)
  
  y = y - dt/dx * ( F_r - F_l )

L = 1.
N = 100
cfl = 0.5
c = 1.
method = 'vanLeer'

x = np.linspace(-0.5*L,0.5*L,N, endpoint=False)
y = np.zeros_like(x)
dx = L/(N-1)
dt = cfl * dx/c
a = dt * c / dx
t = 0.

fig, ax = plt.subplots(1,1)
ax.grid()
ax.set_xlim(-0.5*L, 0.5*L)
ax.set_ylim(-1.5, 1.5)

init()
line, = ax.plot(x,y)

def update(i):
  global x, y, t
  
  # step_Upwind(i)
  # step_LxW()
  step()

  t += dt
  
  line.set_ydata(y) 
  
  return line, 

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()
