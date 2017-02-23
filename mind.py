# number of particles
# number density
# time step
# cutoff radius
# Number of steps
# reduced unit: epsilon = sigma = Kb = 1

import time
import numpy as np
from numba import jit

def initialize(N, L, rx, ry, rz):
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
@jit
def potential_energy(N, L, rc2, rx, ry, rz, fx, fy, fz):
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
    e = 0.0
    for i in range(N):
        vx[i] += 0.5 * dt * fx[i]
        vy[i] += 0.5 * dt * fy[i]
        vz[i] += 0.5 * dt * fz[i]
        e += vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
    e *= 0.5
    return e

def output_xyz(N, z, rx, ry, rz):
    with open('output.xyz', 'a+') as f:
        f.write(str(N) + '\n\n')
        for i in range(N):
            f.write('{:s} {:.8f} {:.8f} {:.8f}'.format('Pu', rx[i], ry[i], rz[i]))
            f.write('\n')

if (__name__ == '__main__'):
    z = 18
    L = 7.55952
    N = 216
    dt = 0.001
    dt2 = dt * dt
    rc2 = 1.e20
    nSteps = 100
    T = 1.0

    rx = np.zeros(N)
    ry = np.zeros(N)
    rz = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    vz = np.zeros(N)
    fx = np.zeros(N)
    fy = np.zeros(N)
    fz = np.zeros(N)

    initialize(N, L, rx, ry, rz)
    output_xyz(N, z, rx, ry, rz)

    print('Setting up run ...')
    print('-' * 70)

    for s in range(nSteps):
        for i in range(N):
            rx[i] += vx[i] * dt + 0.5 * dt2 * fx[i]
            ry[i] += vy[i] * dt + 0.5 * dt2 * fy[i]
            rz[i] += vz[i] * dt + 0.5 * dt2 * fz[i]

            vx[i] += 0.5 * dt * fx[i]
            vy[i] += 0.5 * dt * fy[i]
            vz[i] += 0.5 * dt * fz[i]

            #Periodic boundary conditions
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

        PE = potential_energy(N, L, rc2, rx, ry, rz, fx, fy, fz)
        KE = kinetic_energy(N, dt, vx, vy, vz, fx, fy, fz)
        TE = PE + KE

        print('Step: {:9d} PE = {:12.4f} KE = {:12.4f} TE  = {:12.4f}'.format(s+1, PE, KE, TE))

        output_xyz(N, z, rx, ry, rz)

    print('-' * 70)
    print('End of simulation! :)')
