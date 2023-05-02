import pygame
import OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy
from utils import fitRotationL1, rotdata, ArapConstrainedSolve 
import pickle
import igl
import trimesh
from pydrake.all import StartMeshcat
from vis import plot_mesh, plot_vectors, plot_point

def cotangentWeights(v,t):
    L = csr_matrix((len(v), len(v)))
    for i, tri in enumerate(t):
        for offset in range(3):
            p_idx = tri[offset % 3]
            q_idx = tri[(offset + 1) % 3]
            r_idx = tri[(offset + 2) % 3]
            p = v[p_idx, :]
            q = v[q_idx, :]
            r = v[r_idx, :]
            v_qp = p-q
            v_qr = r-q
            v_rp = p-r
            v_rq = -v_qr
            cotalph = np.dot(v_qp,v_qr)/np.linalg.norm(np.cross(v_qp,v_qr))#sqrt(norm(v_qp)^2 * norm(v_qr)^2 - dot(v_qp,v_qr)^2)
            cotbeta = np.dot(v_rp,v_rq)/np.linalg.norm(np.cross(v_rp,v_rq))#sqrt(norm(v_rp)^2 * norm(v_rq)^2 - dot(v_rp,v_rq)^2)
            L[p_idx, p_idx] += cotalph + cotbeta
            L[p_idx, r_idx] += -cotalph 
            L[p_idx, q_idx] += -cotbeta 

    return L





V, _, _, F, _, _ = igl.read_obj('src/meshes/bunny.obj')
#dummy example
# n = 6
# r = 10
# angs = np.linspace(0, 2*np.pi, n+1)[:-1]
# V = [[0,0,0]]
# for a in angs:
#     V.append([r*np.cos(a), np.sin(a), 0])
# V = np.array(V)
# F = [[0,i+1, i+2] for i in range(6)]
# F[-1] = [0,6,1]
# F = np.array(F, dtype=np.int32)
# from pydrake.all import RollPitchYaw, RotationMatrix
# R = RotationMatrix(RollPitchYaw(np.array([np.pi/6, np.pi/2, np.pi/3])))
# for idx, u in enumerate(U):
#     U[idx, :] = R@u

L = 0.5*(igl.cotmatrix(V,F))
#L2 = cotangentWeights(V,F)
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


# bc = U[vpin, :]
# b = np.array(vpin, dtype=np.int32)
# Aeq = csr_matrix((1, rotdata.L.shape[1]))
# Beq = np.zeros((1,3))
# Aeq[0, vpin] = 0
# Beq[0, :] = 0

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
    # Rcol = np.reshape(np.transpose(RAll, (2, 0, 1)), (len(V) * 3 * 3, 1))
    # Bcol = RHS_ARAP@Rcol
    # Btot = np.reshape(Bcol, (int(Bcol.size / 3), 3))
    # for co in range(3):
    #     success, A = igl.min_quad_with_fixed(rotdata.L, Btot[:,co], b, bc[:,co], Aeq, Beq[:,co], True )
    #     U[:, co] = A
    #rotdata.N = igl.per_vertex_normals(U, F)
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

#rotnorm = np.array([(R@n).squeeze() for R,n in zip(RAll.permutedims(), rotdata.N)])

#plot_vectors(meshcat_handle, N[::10,:], 0.005*U[::10,:], multiplier = 0.05)
while True:
    pass
#mesh = trimesh.Trimesh(vertices=U, faces = F)
#mesh.show()
# #COMPUTE L
# COMPUTE_L = False
# scene = pywavefront.Wavefront('src/meshes/bunny_lowpoly.obj', collect_faces=True)

# scene_box = (scene.vertices[0], scene.vertices[0])
# for vertex in scene.vertices:
#     min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
#     max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
#     scene_box = (min_v, max_v)

# scene_size     = [scene_box[1][i]-scene_box[0][i] for i in range(3)]
# max_scene_size = max(scene_size)
# scaled_size    = 5
# scene_scale    = [scaled_size/max_scene_size for i in range(3)]
# scene_trans    = [-(scene_box[1][i]+scene_box[0][i])/2 for i in range(3)]


# scene.mesh_list

# V = np.array(scene.vertices)
# F = np.array(scene.mesh_list[0].faces)

# #compute barycentric areas
# #compute cotangent weights

# #compute barycentric area
# def compute_area(v, t):
#     verts = v[t, :]
#     b = np.linalg.norm(verts[1,:]-verts[0,:])
#     #gramschmidt
#     e2 = (verts[1,:]- verts[0,:])/b
#     v3 = verts[2,:]- verts[0,:]
#     e3= v3 - np.dot(v3,e2)*e2
#     e3 = e3/np.linalg.norm(e3)
#     h = np.dot(v3, e3)
#     return 0.5*b*h

# def barycentricArea(v,t):
#     baryarea = np.zeros(len(v))
#     for i in range(len(v)):
#         tris_of_interest = np.where(t == i)[0]
#         ba = 0
#         for tidx in tris_of_interest:
            
#             ba += compute_area(v, t[tidx, :])/3
#         baryarea[i] = ba
#     return baryarea



# def computeVertexNormals(v,t):
#     normals = np.zeros((len(v), 3))
#     for i in range(len(v)):
#         tris_of_interest = np.where(t == i)[0]
#         for tidx in tris_of_interest:
#             tri = t[tidx]
#             a = v[tri[0], :]
#             b = v[tri[1], :]
#             c = v[tri[2], :]
#             v1 = b - a
#             v2 = c - a
#             n = np.cross(v1, v2)
#             normals[i, :] += n
#     normals = normals/np.linalg.norm(normals, axis =1 ).reshape(-1,1) 
#     return normals

# barycentricAreas = barycentricArea(V, F)
# vertex_normals = computeVertexNormals(V,F)

# if COMPUTE_L:
#     cotL = cotangentWeights(V,F)
#     with open('src/meshes/bunny_cotLap.bin', 'wb') as f:
#         data = cotL.data
#         idxs = cotL.nonzero()
#         pickle.dump([data, idxs], f)
# else:
#         pass
#     #with open('src/meshes/bunny_cotLap.bin', 'rb') as f:
#         #cotL = pickle.load(f)
#         #cotL = csr_matrix((cotL[0], cotL[1]))
# #pickle cotangent weights because takes too long to compute


# #ADMM rotation fitting parameters
# ADMMParams = {}
# ADMMParams["lambda"] = 4e-1 # cubeness
# ADMMParams["rho"] = 1e-4
# ADMMParams["ABSTOL"] = 1e-5
# ADMMParams["RELTOL"] = 1e-3
# ADMMParams["mu"] = 5
# ADMMParams["tao"] = 2
# ADMMParams["maxIter_ADMM"] = 100

# #Meshdata
# Meshdata = {}
# Meshdata["bareas"] = barycentricAreas
# Meshdata["vertexnormlas"] = vertex_normals

# def Model():
#     glPushMatrix()
#     glScalef(*scene_scale)
#     glTranslatef(*scene_trans)

#     for mesh in scene.mesh_list:
#         glBegin(GL_TRIANGLES)
#         for face in mesh.faces:
#             for vertex_i in face:
#                 glVertex3f(*scene.vertices[vertex_i])
#         glEnd()

#     glPopMatrix()

# def main():
#     pygame.init()
#     display = (800, 600)
#     pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
#     gluPerspective(45, (display[0] / display[1]), 1, 500.0)
#     glTranslatef(0.0, 0.0, -10)

#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 quit()

#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_LEFT:
#                     glTranslatef(-0.5,0,0)
#                 if event.key == pygame.K_RIGHT:
#                     glTranslatef(0.5,0,0)
#                 if event.key == pygame.K_UP:
#                     glTranslatef(0,1,0)
#                 if event.key == pygame.K_DOWN:
#                     glTranslatef(0,-1,0)

#         glRotatef(1, 5, 1, 1)
#         glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

#         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
#         Model()
#         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

#         pygame.display.flip()
#         pygame.time.wait(10)

# main()