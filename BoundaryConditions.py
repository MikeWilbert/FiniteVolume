import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
plt.style.use('./style.mplstyle')

import matplotlib.animation as animation
from audio2numpy import open_audio

def init():
  
  global x,y

  y = -np.cos(2.*x * 2.*np.pi/L)

  y[x < 0.] = -1.
  y[x < -0.25*L*0.5] =  1.
  y[x < -0.75*L*0.5] =  -1.
  
def step_FTCS():
  global x, y
  global t, dt, dx, c
  
  y_l = np.roll(y, +1)
  y_r = np.roll(y, -1)
  
  # FTCS
  y = y - a * 0.5 * ( y_r - y_l )
  
def step_LxF():
  global x, y
  global a
  
  y_l = np.roll(y, +1)
  y_r = np.roll(y, -1)
  
  # LxF
  y = 0.5 * ( y_r + y_l ) - a * 0.5 * ( y_r - y_l )
  
def step_LxW():
  global x, y
  global a
  
  y_l = np.roll(y, +1)
  y_r = np.roll(y, -1)
  
  # LxF
  y = y - a * 0.5 * ( y_r - y_l ) + 0.5 * a**2 * ( y_r - 2.* y + y_l )
  
def step_Upwind(i):
  global x, y, t
  global a
  global audio
  
  print(audio[i])
  
  y[0] = audio[i]
  # y[0] = np.sin(2.*np.pi*t)
  
  y_l = np.roll(y, +1)[1:]
  y_m = y[1:]
  
  # Upwind
  y[1:] = y_m - a * ( y_m - y_l )

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

def step_minmod():
  global x, y
  global a, dx, dt, c
  
  y_l = np.roll(y, +1)
  y_r = np.roll(y, -1)
  
  m_j = minmod( (y - y_l), (y_r - y) ) / dx
  
  # minmod
  y = y - a * ( y - y_l ) - 0.5 * a * ( dx - c*dt ) * ( m_j - np.roll(m_j, +1) )
  
def step_MC():
  global x, y
  global a, dx, dt, c
  
  y_l = np.roll(y, +1)
  y_r = np.roll(y, -1)
  
  m_j = minmod3( 2.*(y - y_l), 2.*(y_r - y), 0.5*(y_r-y_l) ) / dx
  
  # minmod
  y = y - a * ( y - y_l ) - 0.5 * a * ( dx - c*dt ) * ( m_j - np.roll(m_j, +1) )

L = 1.
N = 100
# cfl = 0.5
cfl = 1.
c = 1.

x = np.linspace(-0.5*L,0.5*L,N, endpoint=False)
y = np.zeros_like(x)
dx = L/(N-1)
dt = cfl * dx/c
a = dt * c / dx
t = 0.

fp = "./Audio.mp3"
signal, sampling_rate = open_audio(fp)
# audio = 0.5 * ( signal[:,0] + signal[:,1] )
audio = 0.5 * ( signal[::10000,0] + signal[::10000,1] )
audio = audio / np.amax(audio)
print(audio.shape)

fig, ax = plt.subplots(1,1)
ax.grid()
ax.set_xlim(-0.5*L, 0.5*L)
ax.set_ylim(-1.5, 1.5)

init()
line, = ax.plot(x,y)

def update(i):
  global x, y, t
  
  step_Upwind(i)
  # step_LxW()
  # step_MC()

  t += dt
  
  line.set_ydata(y) 
  
  return line, 

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()
