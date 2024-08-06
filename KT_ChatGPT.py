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
  Q[N//2+Ng:] = 2.
  
  # shock
  # Q += 2.
  # Q[N//2+Ng:] = -1.
  
  # shock
  # Q = 0.5-np.sin( x * 2.*np.pi/L)
  # Q = np.sin( x * 2.*np.pi/L + 0.5* 2.*np.pi)
  
  boundary(Q)

def boundary(U):
  
  global bc
  
  # periodic
  if( bc == 'p' ):
    U[0] = U[-2]
    U[1] = U[-3]
    
    U[-1] = U[3]
    U[-2] = U[2]
        
    return
  
  # outflow
  if( bc == 'o' ):
    U[ 0] = U[2]
    U[ 1] = U[2]
    
    U[-1] = U[-3]
    U[-2] = U[-3]

    return

def flux( u ):
  
  return ( 0.5 * u**2 )

def num_flux( u_p, u_m ):
  
  a = np.fmax( np.abs(u_p), np.abs(u_m) )
  
  H = 0.5 * ( flux( u_p ) + flux( u_m ) ) - 0.5 * a * ( u_p - u_m )
  
  return H 

def calc_RHS(U):
  
  global dx, dt, t
  
  dU_L = (U[1:-1] - U[0:-2]) / dx
  dU_R = (U[2:  ] - U[1:-1]) / dx
  
  # Rusanov
  # sigma = np.zeros_like(dU_L)
  # minmod
  sigma = np.where( np.abs(dU_L) < np.abs(dU_R) , dU_L , dU_R )
  sigma = np.where( dU_L*dU_R > 0., sigma ,0. )

  u_plus  = U[ 2:-1 ] - 0.5*dx * sigma[1:  ]
  u_minus = U[ 1:-2 ] + 0.5*dx * sigma[ :-1]
  
  H = num_flux( u_plus, u_minus )
  
  return ( - ( H[1:] - H[:-1] )/dx )

def step():
    global Q, dt, t

    dt = calc_dt()
    
    # Euler
    RHS_1 = calc_RHS(Q)
    Q[2:-2] = Q[2:-2] + dt * RHS_1
    boundary(Q)
    
    # # Heun's method
    # Q_1 = np.copy(Q)

    # RHS_1 = calc_RHS(Q)
    # Q_1[2:-2] = Q[2:-2] + dt * RHS_1
    # boundary(Q_1)

    # RHS_2 = calc_RHS(Q_1)
    # Q[2:-2] += 0.5 * dt * (RHS_1 + RHS_2)
    # boundary(Q)

    t += dt

def calc_dt():
    global Q
    tau = cfl * dx / 2.
    return tau

L = 2.*np.pi
N = 200
cfl = 0.45
bc = 'p'

Ng = 2

dx = L/(N-1)
t = 0.
dt = 0.

x = np.linspace(-0.5*L-Ng*dx,0.5*L+Ng*dx,N+2*Ng, endpoint=True)
Q = np.zeros( N+2*Ng )

fig, ax1 = plt.subplots(1,1, figsize=(10, 20))
ax1.set_xlim(-0.5*L, 0.5*L)
ax1.set_ylim(-1.5, 2.5)

init()

# while (t < 2.):
  
#   step()

line_1, = ax1.plot(x[Ng:-Ng], Q[Ng:-Ng], c = 'k', marker='*')
# line_1, = ax1.plot(x[Ng:-Ng], Q[Ng:-Ng], c = 'xkcd:ocean blue', marker='*')
fill_1 = ax1.fill_between(x[Ng:-Ng], Q[Ng:-Ng], -1.5)

def update(i):
  global Q, t
  global line_1, line_2, fill_1, ax1
  
  step()
  
  title = "time = {:.2f}".format(t)
  fig.suptitle(title)
  
  line_1.set_ydata( Q[Ng:-Ng] )  

  fill_1.remove()
  fill_1 = ax1.fill_between(x[Ng:-Ng], Q[Ng:-Ng], -1.5, facecolor = 'xkcd:bright blue')
  
  return [line_1 , fill_1]

anim = animation.FuncAnimation( fig=fig, func=update, frames = 40, interval = 30 )

plt.show()
