import numpy as np
import meshcat
from pydrake.all import TriangleSurfaceMesh, Rgba, SurfaceTriangle, Sphere

def plot_mesh(meshcat_handle, V, F, rgba=Rgba(.87, .6, .6, 1.0)):
    tri_drake = [SurfaceTriangle(*t) for t in F]
    meshcat_handle.SetObject("/collision_constraint",
                            TriangleSurfaceMesh(tri_drake, V),
                            Rgba(1, 0, 0, 1), wireframe=False)
    
from pydrake.all import Cylinder, RotationMatrix, RollPitchYaw, RigidTransform

def plot_vector(meshcat_handle, name, vector, location, multiplier = 1.0):
    dir = vector/np.linalg.norm(vector)
    R = RotationMatrix(RollPitchYaw([np.arcsin(-dir[1]),np.arctan2(dir[0], dir[2]), 0]))# RotationMatrix(RollPitchYaw([0,np.arcsin(-dir[1]), np.arctan2(dir[0], dir[2])]))
    offset = np.array([0,0,1]) *  np.linalg.norm(vector)*multiplier/2
    offset_rot = R@offset
    TF = RigidTransform(R = R, p = location.squeeze()+offset_rot.squeeze())
    cyl  = Cylinder(0.001, np.linalg.norm(vector)*multiplier) 
    meshcat_handle.SetObject("/vectors/"+name, 
                             cyl,
                             Rgba(0, 1, 0, 1),
                             )

    meshcat_handle.SetTransform("/vectors/"+name, 
                             TF)                             

def plot_vectors(meshcat_handle, V, Locs, multiplier = 1.0):
    for i,(v,l) in enumerate(zip(V, Locs)):
        plot_vector(meshcat_handle, f"n_{i}", v, l, multiplier)

def plot_point(meshcat_handle, loc,r):
    n  = "/poi/"+str(np.random.rand())
    meshcat_handle.SetObject(n, 
                             Sphere(r),
                             Rgba(0, 0, 1, 1),
                             )

    meshcat_handle.SetTransform(n, 
                             RigidTransform(p = loc.squeeze()))    
# def plot_coordinate_frame(meshcat_handle, name, rotation, location):
#     # Create a Meshcat visualizer
#     #vis = meshcat.Visualizer()

#     # Create coordinate axes
#     axis_length = 1.0
#     x_axis = Cylinder(0.02, axis_length)  # X-axis (red)
#     y_axis = Cylinder(0.02, axis_length)  # Y-axis (green)
#     z_axis = Cylinder(0.02, axis_length)  # Z-axis (blue)
#     rot = RotationMatrix(rotation)
#     tf_x = RigidTransform(rotation=rot*RotationMatrix(RollPitchYaw([90,0,0])), translation=[axis_length / 2+location[0], location[1], location[2]])
#     tf_y = RigidTransform(rotation=rot*RotationMatrix(RollPitchYaw([0,90,0])), translation=[location[0], axis_length / 2+ location[1], location[2]])
#     tf_z = RigidTransform(rotation=rot, translation=[location[0], location[1], axis_length / 2 + location[2]])
#     meshcat_handle.SetObject("/"+name+"x", 
#                              x_axis,
#                              Rgba(1, 0, 0, 1),
#                              )
#     meshcat_handle.SetTransform("/"+name+"x", 
                             
#                              Rgba(1, 0, 0, 1),
#                              )

#     # Set colors
#     x_axis.set_transform()
#     y_axis.set_transform()
#     z_axis.set_transform()

#     x_axis.set_color(1, 0, 0)
#     y_axis.set_color(0, 1, 0)
#     z_axis.set_color(0, 0, 1)

#     # Add the axes to the visualizer
#     meshcat_handle.set_object(x_axis)
#     meshcat_handle.set_object(y_axis)
#     meshcat_handle.set_object(z_axis)

#     # Display the visualizer
#     vis.jupyter_cell()

# # Example usage:
# plot_coordinate_frame()