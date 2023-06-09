import numpy as np
from pydrake.all import Rgba, Sphere, SurfaceTriangle, TriangleSurfaceMesh


def plot_mesh(meshcat_handle, V, F, rgba=Rgba(1, 0, 0, 1), name=""):
    tri_drake = [SurfaceTriangle(*t) for t in F]
    if len(name) == 0:
        name = "/mesh_name/" + str(np.random.rand())
    else:
        name = "/mesh_name/" + name
    meshcat_handle.SetObject(
        name, TriangleSurfaceMesh(tri_drake, V), rgba, wireframe=False
    )


from pydrake.all import Cylinder, RigidTransform, RollPitchYaw, RotationMatrix


def plot_vector(
    meshcat_handle,
    name,
    vector,
    location,
    multiplier=1.0,
    radius=0.001,
    rgba=Rgba(0, 1, 0, 1),
):
    dir = vector / np.linalg.norm(vector)
    R = RotationMatrix(
        RollPitchYaw([np.arcsin(-dir[1]), np.arctan2(dir[0], dir[2]), 0])
    )  # RotationMatrix(RollPitchYaw([0,np.arcsin(-dir[1]), np.arctan2(dir[0], dir[2])]))
    offset = np.array([0, 0, 1]) * np.linalg.norm(vector) * multiplier / 2
    offset_rot = R @ offset
    TF = RigidTransform(R=R, p=location.squeeze() + offset_rot.squeeze())
    cyl = Cylinder(radius, np.linalg.norm(vector) * multiplier)
    meshcat_handle.SetObject(
        "/vectors/" + name,
        cyl,
        rgba,
    )

    meshcat_handle.SetTransform("/vectors/" + name, TF)


def plot_vectors(
    meshcat_handle, V, Locs, multiplier=1.0, radius=0.001, rgba=Rgba(0, 1, 0, 1)
):
    for i, (v, l) in enumerate(zip(V, Locs)):
        plot_vector(
            meshcat_handle, f"n_{i}", v, l, multiplier, radius=radius, rgba=rgba
        )


def plot_point(meshcat_handle, loc, r, rgba=Rgba(0, 0, 1, 1)):
    n = "/poi/" + str(np.random.rand())
    meshcat_handle.SetObject(
        n,
        Sphere(r),
        rgba,
    )

    meshcat_handle.SetTransform(n, RigidTransform(p=loc.squeeze()))
