# number of particles
# number density
# time step
# cutoff

import time
import numpy as np

def initialize():
    pass

def total_energy(N, L, rc2, rx, ry, rz, fx, fy, fz):
    hL = L / 2.0
    e = 0.0
    for i in range(N-1):
        for j in range(N):
            dx = rx[i] - rx[j]
            dy = ry[i] - ry[j]
            dz = rz[i] - rz[i]

            dx -= L if dx > hL else -L
            dy -= L if dy > hL else -L
            dz -= L if dz > hL else -L

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

def run():
    pass

def compute():
    pass

def update():
    pass

def md():
    pass

def output_xyz(N, rx, ry, rz):
    with open('output.xyz', 'a+') as f:
        f.write(str(N) + '\n')
        f.write('TimeStep: \n')
        for i in range(N):
            f.write('{:d} {:.8f} {:.8f} {:.8f}'.format(1, rx[i], ry[i], rz[i]))
            f.write('\n')

if (__name__ == '__main__'):
    rx=[1,2,3]
    ry=[1,2,3]
    rz=[1,2,3]
    output_xyz(3,rx,ry,rz)
