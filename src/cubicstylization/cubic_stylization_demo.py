import igl
import numpy as np
from pydrake.all import StartMeshcat
from vis import plot_mesh, plot_vectors, plot_point
from functools import partial
from cubic_stylization import CubiclyStylize
import pickle
import trimesh

#load mesh

#V, _, _, F, _, _ = igl.read_obj('src/meshes/bunny.obj')
mesh = trimesh.load_mesh("src/meshes/latest_mesh.stl")
mesh = sorted(mesh.split(), key=lambda m: m.faces.shape[0])[-1]
V = mesh.vertices
F = mesh.faces
Vm = np.mean(V, axis = 0).reshape(1,-1)
V = V-Vm

#mesh = trimesh.Trimesh(vertices = V, faces = F)

def plotting_function(U, F, Vpin_loc, meshcat_handle):
    for v in Vpin_loc:
        plot_point(meshcat_handle, 0.005*v, r = 0.01)
    plot_mesh(meshcat_handle, 0.005*U, F, name='mesh')

#start meshcat, and create plotting handle
meshcat = StartMeshcat()
plotting_handle = partial(plotting_function, meshcat_handle = meshcat)

#constraint points
vpin = [10, 15, 30]
cons = V[vpin,:]
#cons[1, 0] -=18 

#apply cubic stylization
U, RAll = CubiclyStylize(V,
                         F,
                         V_pinned_index_list=vpin,
                         V_pinned_locations=cons,
                         max_alternations=20,
                         lambda_= 0.25,
                         ADMM_iters = 100,
                         plotting_handle=plotting_handle
                         )


with open('src/meshes/mic_stylized.bin', 'wb') as f:
        data = [V+Vm, U+Vm, F, RAll]
        pickle.dump(data, f)

while True:
    pass

