import pickle
from pathlib import Path

import hydra
import numpy as np
import trimesh
from omegaconf import DictConfig
from trimesh import Trimesh

from .cubic_stylization import CubiclyStylize


@hydra.main(
    version_base=None,
    config_path="../../config/cubicstylization",
    config_name="main",
)
def stylize(cfg: DictConfig):
    mesh = trimesh.load_mesh(cfg.input)
    mesh = sorted(mesh.split(), key=lambda m: m.faces.shape[0])[-1]
    V = mesh.vertices
    F = mesh.faces
    Vm = np.mean(V, axis=0).reshape(1, -1)
    V = V - Vm

    # constraint points (random for now)
    vpin = [10, 15, 30]
    cons = V[vpin, :]

    # apply cubic stylization
    U, RAll = CubiclyStylize(
        V,
        F,
        V_pinned_index_list=vpin,
        V_pinned_locations=cons,
        max_alternations=20,
        lambda_=cfg.lambda_,
        ADMM_iters=100,
    )

    deformed = Trimesh(U + Vm, F)
    Path(cfg.output).parent.mkdir(exist_ok=True, parents=True)
    deformed.export(cfg.output)
    with open(cfg.output_package, "wb") as f:
        pickle.dump((U + Vm, V + Vm, F, RAll), f)


if __name__ == "__main__":
    stylize()
