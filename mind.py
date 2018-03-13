################################################################################
#              ,--.   ,--. ,--.             ,--.                               #
#              |   `.'   | `--' ,--,--,   ,-|  |                               #
#              |  |'.'|  | ,--. |      \ ' .-. |                               #
#              |  |   |  | |  | |  ||  | \ `-' |                               #
#              `--'   `--' `--' `--''--'  `---'                                #
#                                                                              #
# ** A Minimal Lennard-Jones Fluid Molecular Dynamics Python Program **        #
#                                                                              #
#                                                                              #
#                                                                              #
#  Author: Pu Du                                                               #
# Website: pudu.io                                                             #
#   Email: pudugg@gmail.com                                                    #
################################################################################

import time
import numpy as np
from numba import jit

def initialize(N, L, rx, ry, rz):
    """put N particles in a box"""
    n3 = int(N ** (1 / 3.)) + 1
    iix = iiy = iiz = 0
    for i in range(N):
        rx[i] = (iix + 0.5) * L / n3
        ry[i] = (iiy + 0.5) * L / n3
        rz[i] = (iiz + 0.5) * L / n3
        iix += 1
        if iix == n3:
            iix = 0
            iiy += 1
            if iiy == n3:
                iiy = 0
                iiz += 1

def velocity_verlet(dt, rx, ry, rz, vx, vy, vz, i):
    """verloctiy verlet algorithm"""
    dt2 = dt * dt
    rx[i] += vx[i] * dt + 0.5 * dt2 * fx[i]
    ry[i] += vy[i] * dt + 0.5 * dt2 * fy[i]
    rz[i] += vz[i] * dt + 0.5 * dt2 * fz[i]

    vx[i] += 0.5 * dt * fx[i]
    vy[i] += 0.5 * dt * fy[i]
    vz[i] += 0.5 * dt * fz[i]

def wrap_into_box(L, rx, ry, rz, i):
    """wrap the coordinates"""
    if rx[i] < 0.0:
        rx[i] += L
    if rx[i] > L:
        rx[i] -= L
    if ry[i] < 0.0:
        ry[i] += L
    if ry[i] > L:
        ry[i] -= L
    if rz[i] < 0.0:
        rz[i] += L
    if rz[i] > L:
        rz[i] -= L

@jit
def potential_energy(N, L, rc2, rx, ry, rz, fx, fy, fz):
    """calculate the potential energy"""
    fx.fill(0)
    fy.fill(0)
    fz.fill(0)

    hL = L / 2.0
    e = 0.0
    for i in range(N-1):
        for j in range(i+1, N):
            dx = rx[i] - rx[j]
            dy = ry[i] - ry[j]
            dz = rz[i] - rz[j]

            if dx > hL:
                dx -= L
            if dx < -hL:
                dx += L
            if dy > hL:
                dy -= L
            if dy < -hL:
                dy += L
            if dz > hL:
                dz -= L
            if dz < -hL:
                dz += L

            r2 = dx * dx + dy * dy + dz * dz

            if r2 < rc2:
                r6i = 1.0 / (r2 * r2 * r2)
                e += 4 * (r6i * r6i - r6i)
                f = 48 * (r6i * r6i - 0.5 * r6i)
                fx[i] += dx * f / r2
                fx[j] -= dx * f / r2
                fy[i] += dy * f / r2
                fy[j] -= dy * f / r2
                fz[i] += dz * f / r2
                fz[j] -= dz * f / r2
    return e

@jit
def kinetic_energy(N, dt, vx, vy, vz, fx, fy, fz):
    """calculate the kinetic energy"""
    e = 0.0
    for i in range(N):
        vx[i] += 0.5 * dt * fx[i]
        vy[i] += 0.5 * dt * fy[i]
        vz[i] += 0.5 * dt * fz[i]
        e += vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
    e *= 0.5
    return e

def berendsen_thermostat(N, dt, KE, vx, vy, vz):
    """berendsen thermostat algorithm"""
    lamb = np.sqrt(1 + dt / tau * (T / (2.0 * KE / 3.0 / N) - 1.0))
    for i in range(N):
        vx[i] *= lamb
        vy[i] *= lamb
        vz[i] *= lamb

def thermostat(KE, T, N, vx, vy, vz):
    """velocity scaling algorithm"""
    t = KE / N * 2. / 3.
    fac = np.sqrt( T / t)
    for i in range(N):
        vx[i] *= fac
        vy[i] *= fac
        vz[i] *= fac

def output_welcome(N, L, dt, nSteps, T):
    """print welcome information"""
    mind = '''
              ,--.   ,--. ,--.             ,--.
              |   `.'   | `--' ,--,--,   ,-|  |
              |  |'.'|  | ,--. |      \ ' .-. |
              |  |   |  | |  | |  ||  | \ `-' |
              `--'   `--' `--' `--''--'  `---'
          '''
    print(mind)
    print('** A Minimal Lennard-Jones Fluid Molecular Dynamics Python Program **')
    print('\nSystem information:\n')
    print('               ** ALL UNITS ARE IN REDUCED UNITS **\n')
    print('                 Simulation type:\tNVT')
    print('  Number of particles in the box:\t{:d}'.format(N))
    print('                      Box length:\t{:f}'.format(L))
    print('                       Time step:\t{:f}'.format(dt))
    print('           Total number of steps:\t{:d}'.format(nSteps))
    print('              Target temperature:\t{:f}'.format(T))
    print('\nOutput format:\n')
    print ( 'Step             Potential        Kinetic          Total' )
    print ( '                 Energy PE        Energy KE        Energy TE\n' )

def output_thermo(s, PE, KE, TE):
    """print thermo information"""
    print('Step: {:9d} PE = {:12.4f} KE = {:12.4f} TE  = {:12.4f}'.format(s+1, PE, KE, TE))

def output_xyz(N, rx, ry, rz):
    """xyz output"""
    with open('output.xyz', 'a+') as f:
        f.write(str(N) + '\n\n')
        for i in range(N):
            f.write('{:s} {:.8f} {:.8f} {:.8f}'.format('Pu', rx[i], ry[i], rz[i]))
            f.write('\n')

def output_end(t_start, t_end):
    """print time information"""
    print('-' * 70)
    print ('Total looping time = {:.2f} seconds.'.format(t_end - t_start))
    art = '''
                           .
                          ":"
                        ___:____     |"\/"|
                      ,'        `.    \  /
                      |  O        \___/  |
                    ~^~^~^~^~^~^~^~^~^~^~^~^~
           '''
    print(art)

def mdrun(N, L, rc2, dt, nSteps, T, rx, ry, rz, vx, vy, vz, fx, fy, fz):
    """main MD function"""
    print('-' * 70)
    for s in range(nSteps):
        for i in range(N):
            velocity_verlet(dt, rx, ry, rz, vx, vy, vz,i)
            wrap_into_box(L, rx, ry, rz, i)

        PE = potential_energy(N, L, rc2, rx, ry, rz, fx, fy, fz)
        KE = kinetic_energy(N, dt, vx, vy, vz, fx, fy, fz)
        TE = PE + KE

        berendsen_thermostat(N, dt, KE, vx, vy, vz)

        output_thermo(s, PE, KE, TE)
        output_xyz(N, rx, ry, rz)

if (__name__ == '__main__'):
    ############################################
    # parameters can be changed (reduced unit)
    # L : box length
    # N : number of particles
    # dt : time step
    # rc2 : squared cutoff distance
    # nSteps : number of steps of simulation
    # T : temperature
    L = 7.55952
    N = 216
    dt = 0.001
    rc2 = 1.e20
    nSteps = 10000
    T = 1.0
    tau = 0.1
    Tdamp = 1
    ############################################

    rx = np.zeros(N)
    ry = np.zeros(N)
    rz = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    vz = np.zeros(N)
    fx = np.zeros(N)
    fy = np.zeros(N)
    fz = np.zeros(N)


    output_welcome(N, L, dt, nSteps, T)
    initialize(N, L, rx, ry, rz)
    output_xyz(N, rx, ry, rz)

    t_start = time.clock()
    mdrun(N, L, rc2, dt, nSteps, T, rx, ry, rz, vx, vy, vz, fx, fy, fz)
    t_end = time.clock()
    output_end(t_start, t_end)
