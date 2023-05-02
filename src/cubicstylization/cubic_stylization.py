import pygame
import OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.sparse import csr_matrix
import scipy
from utils import fitRotationL1, rotdata, ArapConstrainedSolve 
import igl

from pydrake.all import StartMeshcat
from vis import plot_mesh, plot_vectors, plot_point

V, _, _, F, _, _ = igl.read_obj('src/meshes/bunny.obj')

L = 0.5*(igl.cotmatrix(V,F))
VA = igl.massmatrix(V,F,igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
rotdata.F = F.copy()
rotdata.L = csr_matrix(L.copy())
rotdata.V = V.copy()
rotdata.N = igl.per_vertex_normals(V,F)
rotdata.VA = VA
rotdata.lambda_ = 0.2
RHS_ARAP = igl.arap_rhs(V,F,3, energy = igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)

vpin = [10, 2335, 30, 4100]
tol = 1e-3
iter = 20
objHis = []
UHis = np.zeros((len(V), 3, iter))

#reduce laplacian
cols = [i for i in range(len(V))]
for v in vpin:
    cols.remove(v)
L_red = scipy.sparse.lil_matrix(L[:,cols])
L_red = scipy.sparse.lil_matrix(L_red[cols,:])
U = V.copy()
cons = V[vpin,:]
cons[1, 0] -=18 
U[vpin, :] = cons
meshcat_handle = StartMeshcat()

for v in vpin:
    plot_point(meshcat_handle, 0.005*U[v,:], r = 0.01)
plot_mesh(meshcat_handle, 0.005*U, F)

for it in range(iter):
    RAll, val, rotdata = fitRotationL1(U, rotdata)
    # save optimization info
    objHis.append(val)
    UHis[:, :, it] = U
    # global step
    UPre = U
    U = ArapConstrainedSolve(L, L_red, RAll, V, cons, vpin)

    # stopping criteria
    dU = np.sqrt(np.sum((U - UPre) ** 2, axis=1))
    dUV = np.sqrt(np.sum((U - V) ** 2, axis=1))
    if np.max(dUV) ==0:
        print('converged')
        break
    reldV = np.max(dU) / np.max(dUV)
    print('iter: %d, objective: %d, reldV: %d' % (it, val, reldV))
    plot_mesh(meshcat_handle, 0.005*U, F)
    if reldV < tol:
       break

N = igl.per_vertex_normals(U,F)
nvecs = 10
idx = np.random.permutation(np.arange(len(V)))[:nvecs]
plot_vectors(meshcat_handle, N[idx,:], 0.005*U[idx,:], multiplier = 0.05)
while True:
    pass
