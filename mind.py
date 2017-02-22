# number of particles
# number density
# time step
# cutoff radius
# Number of steps

import time
import numpy as np

def initialize():
    n3 = int(n ** (1 / 3.))
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

def total_energy(N, L, rc2, rx, ry, rz, fx, fy, fz):
    hL = L / 2.0
    e = 0.0
    for i in range(N-1):
        for j in range(N):
            dx = rx[i] - rx[j]
            dy = ry[i] - ry[j]
            dz = rz[i] - rz[i]

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

def compute():
    pass

def update():
    pass

def md():
    pass

def thermo():
    pass

def output_xyz(N, rx, ry, rz):
    with open('output.xyz', 'a+') as f:
        f.write(str(N) + '\n')
        f.write('TimeStep: \n')
        for i in range(N):
            f.write('{:d} {:.8f} {:.8f} {:.8f}'.format(1, rx[i], ry[i], rz[i]))
            f.write('\n')

def run():
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

        PE = total_energy(N, L, rc2, rx, ry, rz, fx, fy, fz)

        KE = 0.0
        for i in range(N):
            vx[i] += 0.5 * dt * fx[i]
            vy[i] += 0.5 * dt * fy[i]
            vz[i] += 0.5 * dt * fz[i]

            KE += vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
        KE *= 0.5
        TE = PE + KE
        print('steps: {:d}'.format())
        print('\n')
        output_xyz(N, rx, ry, rz)

if (__name__ == '__main__'):
    '''
    rx=[1,2,3]
    ry=[1,2,3]
    rz=[1,2,3]
    output_xyz(3,rx,ry,rz)
    '''
    #TODO: initializa
    #output initial positions
    output_xyz(N, rx, ry, rz)

    TE0 = total_energy(N, L, rc2, rx, ry, rz, fx, fy, fz)

    #TODO: run section
    '''
