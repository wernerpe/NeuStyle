import torch
import numpy as np
import igl
import pickle
from src.cubicstylization.vis import plot_vectors, plot_point, plot_mesh
from pydrake.all import StartMeshcat, Rgba
import trimesh
from scipy.sparse import csr_matrix

with open('src/meshes/bunny_cubified.bin', 'rb') as f:
    data = pickle.load(f)
    V, U, F, RAll = tuple(data)


# convert straight rays in deformed space to piecewise sampling locations for  
def map_rays_to_neus(ray_origins, ray_directions, V, U, F, n_samples = 21, scale= 2.0):
    '''
    take in rays in deformed space and map them to the original neus, by splitting them
    into linesegments around the surface, where sdf = 0 
    
    input:
    ray_origins: |R|x3 ray origin locations
    ray_directions: |R|x3 ray directions, unit vectors expected
    V: |V|x3 vertices of original mesh
    U: |V|x3 vertices of stylized mesh
    F: |F|x3 faces of mesh

    output:
    sampling_points_mat |R|x|S|x3 matrix indexed by ray returnin an |s|x3 matrix of samples near the surface
    sampling_points_mat_undef |R|x|S|x3 matrix indexed by ray returnin an |s|x3 matrix
                                 of samples near the surface of undeformed mesh
    '''
    assert n_samples%2 == 1
    if torch.zeros((3,3)).device == torch.device('cpu'):
        r_o = ray_origins.numpy() 
        r_d = ray_directions.numpy()
        was_GPU = False
    else:
        r_o = ray_origins.cpu().numpy() 
        r_d = ray_directions.cpu().numpy()
        was_GPU = True
    mesh = trimesh.Trimesh(vertices=U, faces = F)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    idx_tri, idx_ray, locs_int = intersector.intersects_id(r_o, r_d, return_locations = True, multiple_hits = False)
    #surface_rays = []
    #surface_rays_undef = []
    offset_mult = scale*np.linspace(-1,1, n_samples).reshape(-1,1)
    sampling_points_mat = np.zeros((r_o.shape[0], n_samples, 3))
    sampling_points_mat_undef = np.zeros((r_o.shape[0], n_samples, 3))#csr_matrix(shape=(r_o.shape[0], n_samples, 3))
    for tri_idx, ray_idx, intersection in zip(idx_tri, idx_ray, locs_int):
        closest_vert = np.argmin(np.linalg.norm(U-intersection.reshape(1,3), axis=1))
        surface_offsets_along_normal = offset_mult*np.tile(dir[ray_idx, :],(n_samples,1))
        surface_offsets_along_normal_rot_to_orig = (RAll[:,:, closest_vert].T@surface_offsets_along_normal.T).T
        sampling_points = surface_offsets_along_normal_rot_to_orig + V[closest_vert, :].reshape(1,3)
        #surface_rays.append(sampling_points)
        sampling_points_mat[ray_idx, :] = sampling_points
        sampling_points_mat_undef[ray_idx, :] = surface_offsets_along_normal + intersection.reshape(1,3)
    
        #surface_rays_undef.append(surface_offsets_along_normal + intersection.reshape(1,3))
    return sampling_points_mat, sampling_points_mat_undef


    
meshscaling = 0.03
meshcat = StartMeshcat()    
plot_mesh(meshcat, meshscaling*V, F, rgba=Rgba(0,0,1,0.2))
plot_mesh(meshcat, meshscaling*U, F, rgba=Rgba(1,0,0,0.3))

nrays = 3
loc = np.zeros((nrays, 3))
loc[:,0] = 65 
loc[:,2] = 30
dir = np.zeros((nrays, 3))
dir[:, 0] = -1
dir[:, 1] = 0.6 * (np.random.rand(nrays)-0.5) 
dir[:, 2] = 0.3 * np.random.rand(nrays) - 0.4
dir = dir/np.linalg.norm(dir, axis=1).reshape(-1,1)
plot_vectors(meshcat, 5*dir, meshscaling*loc, radius = 0.002, rgba=Rgba(0,1,0,1))
r_o = torch.tensor(loc)
r_d = torch.tensor(dir)

ray_samples_def, ray_samples_undef =  map_rays_to_neus(torch.tensor(loc), torch.tensor(dir), V, U, F)

for r in ray_samples_undef:
     for p in r:
           plot_point(meshcat, meshscaling*p, r = 0.005, rgba = Rgba(1,0,0,1))

for r in ray_samples_def:
     for p in r:
           plot_point(meshcat, meshscaling*p, r = 0.005, rgba = Rgba(0,0,1,1))


mesh = trimesh.Trimesh(vertices=U, faces = F)
intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

idx_tri, idx_ray, locs_int = intersector.intersects_id(loc, dir, return_locations = True, multiple_hits = False)
for l in locs_int:
     plot_point(meshcat, meshscaling*l, r = 0.01, rgba = Rgba(0,1,0.4,1))

N = igl.per_vertex_normals(U,F)
scale = 2.0
n_samps = 21
surface_rays = []
surface_rays_undef = []
offset_mult = scale*np.linspace(-1,1, n_samps).reshape(-1,1)
for tri_idx, ray_idx, intersection in zip(idx_tri, idx_ray, locs_int):
    closest_vert = np.argmin(np.linalg.norm(U-intersection.reshape(1,3), axis=1))
    surface_offsets_along_normal = offset_mult*np.tile(dir[ray_idx, :],(n_samps,1))
    surface_offsets_along_normal_rot_to_orig = (RAll[:,:, closest_vert].T@surface_offsets_along_normal.T).T
    sampling_points = surface_offsets_along_normal_rot_to_orig + V[closest_vert, :].reshape(1,3)
    surface_rays.append(sampling_points)
    surface_rays_undef.append(surface_offsets_along_normal + intersection.reshape(1,3))

for r in surface_rays_undef:
     for p in r:
           plot_point(meshcat, meshscaling*p, r = 0.005, rgba = Rgba(1,0,0,1))

for r in surface_rays:
     for p in r:
           plot_point(meshcat, meshscaling*p, r = 0.005, rgba = Rgba(0,0,1,1))
while True:
     pass
