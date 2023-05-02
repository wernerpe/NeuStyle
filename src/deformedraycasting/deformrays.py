import torch
import numpy as np
import igl
import pickle
from src.cubicstylization.vis import plot_vectors, plot_point, plot_mesh
from pydrake.all import StartMeshcat, Rgba

with open('src/meshes/bunny_cubified.bin', 'rb') as f:
        data = pickle.load(f)
        V, U, F, RAll = tuple(data)


# convert straight rays in deformed space to piecewise sampling locations for  
def map_rays_to_neus(ray_origins, ray_directions, V, U, F):
    '''
    take in rays in deformed space and map them to the original neus, by splitting them
    into linesegments around the surface, where sdf = 0 
    '''
    Normals = igl.per_vertex_normal(U)

    for r_o, r_d in zip(ray_origins, ray_directions):   

        pass
    

meshcat = StartMeshcat()    
plot_mesh(meshcat, 0.03*U, F, rgba=Rgba(1,0,0,0.4))
plot_mesh(meshcat, 0.03*V, F, rgba=Rgba(0,0,1,0.3))

nrays = 10
loc = np.zeros((nrays, 3))
loc[:,0] = 65 
loc[:,2] = 30
dir = np.zeros((nrays, 3))
dir[:, 0] = -1
dir[:, 1] = 0.6 * (np.random.rand(nrays)-0.5) 
dir[:, 2] = 0.3 * np.random.rand(nrays) - 0.4
dir = dir/np.linalg.norm(dir, axis=1).reshape(-1,1)
plot_vectors(meshcat, 5*dir, 0.03*loc, radius = 0.002, rgba=Rgba(0,1,0,1))


r_o = torch.tensor(loc)
r_o = torch.tensor(dir)


