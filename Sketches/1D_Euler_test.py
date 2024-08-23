import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-v0_8-paper')
# plt.style.use('./style.mplstyle')  # Commented out to avoid dependency

import matplotlib.animation as animation
from matplotlib.patches import Ellipse

def boundary(A):
    global Ng
    # Outflow conditions
    A[:, Ng-1] = A[:, Ng]
    A[:, Ng-2] = A[:, Ng-1]
    A[:, -Ng] = A[:, -Ng-1]
    A[:, -Ng+1] = A[:, -Ng]

def primitives2conserved(n, u, p):
    global gamma
    out = np.zeros((3, n.shape[0]))
    out[0, :] = n
    out[1, :] = n * u
    out[2, :] = p / (gamma - 1.) + 0.5 * n * u**2
    return out

def conserved2primitives(A):
    n = A[0, :]
    n = np.maximum(n, 1.e-12)  # Prevent zero or negative density
    u = A[1, :] / n
    p = (A[2, :] - 0.5 * n * u**2) * (gamma - 1.)
    p = np.maximum(p, 1.e-12)  # Prevent zero or negative pressure
    return n, u, p

def init():
    global Q, x, L
    # Sod's problem initial conditions
    n = np.ones_like(x)
    u = np.zeros_like(x)
    p = np.ones_like(x)
    n[x > 0.5 * L] = 0.125
    u[x > 0.5 * L] = 0.
    p[x > 0.5 * L] = 0.1
    Q = primitives2conserved(n, u, p)
    boundary(Q)

def flux(A):
    F = np.zeros_like(A)
    n, u, p = conserved2primitives(A)
    E = A[2, :]
    F[0, :] = u * n
    F[1, :] = u * n * u + p
    F[2, :] = u * (E + p)
    return F

def get_reconstruction(A):
    # Left finite difference stencil (no slope limiting)
    A_l = A[:, Ng-2:-Ng]
    A_c = A[:, Ng-1:-Ng+1]
    sigma_l = (A_c - A_l) / dx

    # Reconstruct states at cell faces
    A_p = A[:, Ng-1:-Ng] + 0.5 * dx * sigma_l[:, :-1]  # Right of i
    A_m = A[:, Ng:-Ng+1] - 0.5 * dx * sigma_l[:, 1:]   # Left of i+1

    return A_p, A_m

def compute_time_step(Q):
    n, u, p = conserved2primitives(Q)
    c = np.sqrt(gamma * p / n)
    max_speed = np.max(np.abs(u) + c)
    dt = cfl * dx / max_speed
    return dt

def step():
    global Q, t, dt, dx
    Q_p, Q_m = get_reconstruction(Q)
    F_p = flux(Q_p)
    F_m = flux(Q_m)
    n_p, u_p, p_p = conserved2primitives(Q_p)
    n_m, u_m, p_m = conserved2primitives(Q_m)
    c_p = np.sqrt(np.maximum(gamma * p_p / n_p, 0.))
    c_m = np.sqrt(np.maximum(gamma * p_m / n_m, 0.))
    a_p = np.abs(u_p) + c_p
    a_m = np.abs(u_m) + c_m
    a = np.maximum(a_p, a_m)
    H = -0.5 * (F_m + F_p - a * (Q_m - Q_p))
    RHS = (H[:, 1:] - H[:, :-1]) / dx
    Q[:, Ng:-Ng] += dt * RHS
    boundary(Q)
    t += dt
    
    print(t)

# Main parameters and initialization
L = 1.
N = 1000
cfl = 0.1
Ng = 2
dx = L / (N - 1)
t = 0.
gamma = 1.4
x = np.linspace(-Ng*dx, L+Ng*dx, N+2*Ng, endpoint=True)
Q = np.zeros((3, N + 2 * Ng))
dt = 0.
init()

# Plotting setup
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
line, = ax.plot(x, Q[0, :], '-')
title = "time = {:.2f} s".format(t)
fig.suptitle(title, fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Density')
for val in [0.263, 0.495, 0.685, 0.850]:
    ax.axvline(val, c='k', ls='--', zorder=0)
ax.set_xlim(0., 1.0)
ax.set_ylim(0., 1.1)

def update(i):

    global dt
  
    global Q, t
    dt = compute_time_step(Q)
    step()
    line.set_ydata(Q[0, :])
    title = "time = {:.2f} s".format(t)
    fig.suptitle(title, fontsize=16)
    return line,

anim = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)

plt.show()
